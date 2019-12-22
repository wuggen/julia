use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

use image::{ImageBuffer, Rgba};

use crate::shaders::julia_comp;
use crate::{ImgDimensions, JuliaContext, JuliaCreationError, JuliaData};

use std::cell::Cell;
use std::fmt::{self, Debug, Formatter};
use std::path::Path;
use std::sync::Arc;

pub type CompDesc = PipelineLayout<julia_comp::Layout>;

pub struct JuliaExport {
    pipeline: Arc<ComputePipeline<CompDesc>>,
    cached_data: Cell<Option<JuliaExportCache>>,
}

struct JuliaExportCache {
    dims: ImgDimensions,
    data: JuliaData,
    input_buffer: Arc<ImmutableBuffer<julia_comp::ty::Data>>,
    image: Arc<StorageImage<Format>>,
    output_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl JuliaExport {
    pub fn new(device: &Arc<Device>) -> Result<JuliaExport, JuliaCreationError> {
        let shader =
            julia_comp::Shader::load(device.clone()).map_err(JuliaCreationError::ShaderLoad)?;
        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
                .map_err(JuliaCreationError::ComputePipelineCreation)?,
        );

        Ok(JuliaExport {
            pipeline,
            cached_data: Cell::new(None),
        })
    }

    fn regen_cache(&self, dims: ImgDimensions, data: &JuliaData, context: &JuliaContext) {
        let shader_data = data.into_shader_data();
        let (input_buffer, future) =
            ImmutableBuffer::from_data(shader_data, BufferUsage::all(), context.queue().clone())
                .unwrap();

        let image = StorageImage::new(
            context.device().clone(),
            Dimensions::Dim2d {
                width: dims.width,
                height: dims.height,
            },
            Format::R8G8B8A8Unorm,
            Some(context.queue().family()),
        )
        .unwrap();

        let output_buffer = CpuAccessibleBuffer::from_iter(
            context.device().clone(),
            BufferUsage::all(),
            (0..dims.width * dims.height * 4).map(|_| 0u8),
        )
        .unwrap();

        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.cached_data.set(Some(JuliaExportCache {
            dims,
            data: *data,
            input_buffer,
            image,
            output_buffer,
        }));
    }

    fn command_buffer(
        &self,
        context: &JuliaContext,
    ) -> AutoCommandBuffer<StandardCommandPoolAlloc> {
        let cache = self.cached_data.take().unwrap();
        let desc_set = Arc::new(
            PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_image(cache.image.clone())
                .unwrap()
                .add_buffer(cache.input_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let cmd_buf =
            AutoCommandBufferBuilder::new(context.device().clone(), context.queue().family())
                .unwrap()
                .dispatch(
                    [cache.dims.width / 8, cache.dims.height / 8, 1],
                    self.pipeline.clone(),
                    desc_set.clone(),
                    (),
                )
                .unwrap()
                .copy_image_to_buffer(cache.image.clone(), cache.output_buffer.clone())
                .unwrap()
                .build()
                .unwrap();

        self.cached_data.set(Some(cache));
        cmd_buf
    }

    pub fn export(
        &self,
        dims: ImgDimensions,
        data: &JuliaData,
        filename: &Path,
        context: &JuliaContext,
    ) {
        let cache_opt = self.cached_data.take();

        match cache_opt {
            None => self.regen_cache(dims, data, context),
            Some(c) => {
                if c.data != *data || c.dims != dims {
                    self.regen_cache(dims, data, context);
                } else {
                    self.cached_data.set(Some(c));
                }
            }
        }

        self.export_core(filename, context);
    }

    fn export_core(&self, filename: &Path, context: &JuliaContext) {
        self.command_buffer(context)
            .execute(context.queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let cache = self.cached_data.take().unwrap();
        let img_contents = cache.output_buffer.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            cache.dims.width,
            cache.dims.height,
            &img_contents[..],
        )
        .unwrap();
        image.save(filename).unwrap();

        drop(img_contents);
        self.cached_data.set(Some(cache));
    }
}

impl Debug for JuliaExport {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let cache = self.cached_data.take();
        let res = f
            .debug_struct("JuliaExport")
            .field("pipeline", &self.pipeline)
            .field(
                "cached_data",
                &match cache {
                    None => String::from("None"),
                    Some(ref c) => format!("{:?}", c.data),
                },
            )
            .finish();
        self.cached_data.set(cache);
        res
    }
}
