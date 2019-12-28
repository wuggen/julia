use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, CommandBufferExecError,
    CommandBufferExecFuture, DispatchError,
};
use vulkano::descriptor::descriptor_set::{
    FixedSizeDescriptorSetsPool, PersistentDescriptorSetBuildError,
};
use vulkano::format::Format;
use vulkano::image::{ImageUsage, Dimensions, ImageCreationError, StorageImage};
use vulkano::memory::DeviceMemoryAllocError;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::{self, GpuFuture, NowFuture};
use vulkano::OomError;

use crate::shaders::julia_comp;
use crate::{CompDesc, JuliaContext, JuliaData};

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::iter;
use std::sync::Arc;

pub struct JuliaImage {
    image: Arc<StorageImage<Format>>,
    buffer_pool: CpuBufferPool<julia_comp::ty::Data>,
    desc_set_pool: RefCell<FixedSizeDescriptorSetsPool<Arc<ComputePipeline<CompDesc>>>>,
}

impl JuliaImage {
    pub fn new(
        context: &JuliaContext,
        dimensions: [u32; 2],
    ) -> Result<JuliaImage, JuliaImageError> {
        let image = create_image(context, dimensions)?;
        let buffer_pool = CpuBufferPool::<julia_comp::ty::Data>::new(
            context.device().clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
        );
        let desc_set_pool = RefCell::new(FixedSizeDescriptorSetsPool::new(
            context.pipeline().clone(),
            0,
        ));

        Ok(JuliaImage {
            image,
            buffer_pool,
            desc_set_pool,
        })
    }

    pub fn draw(
        &self,
        data: JuliaData,
        context: &JuliaContext,
    ) -> Result<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>, JuliaImageError> {
        self.draw_after(sync::now(context.device().clone()), data, context)
    }

    pub fn draw_after<F: GpuFuture>(
        &self,
        future: F,
        data: JuliaData,
        context: &JuliaContext,
    ) -> Result<CommandBufferExecFuture<F, AutoCommandBuffer>, JuliaImageError> {
        let buffer = self.buffer_pool.next(data.into_shader_data())?;
        let desc_set = self
            .desc_set_pool
            .borrow_mut()
            .next()
            .add_image(self.image.clone())
            .unwrap()
            .add_buffer(buffer)
            .unwrap()
            .build()?;

        let [width, height] = self.dimensions();
        let cmd_buf = AutoCommandBufferBuilder::primary_one_time_submit(
            context.device().clone(),
            context.queue().family(),
        )?
        .dispatch(
            [width / 8, height / 8, 1],
            context.pipeline().clone(),
            desc_set,
            (),
        )?
        .build()?;

        Ok(future.then_execute(context.queue().clone(), cmd_buf)?)
    }

    pub fn dimensions(&self) -> [u32; 2] {
        if let Dimensions::Dim2d { width, height } = self.image.dimensions() {
            [width, height]
        } else {
            // image dimensions are only ever 2D by construction
            panic!()
        }
    }

    pub fn image(&self) -> &Arc<StorageImage<Format>> {
        &self.image
    }
}

fn create_image(
    context: &JuliaContext,
    dimensions: [u32; 2],
) -> Result<Arc<StorageImage<Format>>, ImageCreationError> {
    let [width, height] = dimensions;
    let dimensions = Dimensions::Dim2d { width, height };

    StorageImage::with_usage(
        context.device().clone(),
        dimensions,
        Format::R8G8B8A8Unorm,
        ImageUsage {
            transfer_source: true,
            sampled: true,
            storage: true,
            ..ImageUsage::none()
        },
        iter::once(context.queue().family()),
    )
}

#[derive(Debug, Clone)]
pub enum JuliaImageError {
    VkImageErr(ImageCreationError),
    VkAllocErr(DeviceMemoryAllocError),
    VkDescSetErr(PersistentDescriptorSetBuildError),
    VkOomErr(OomError),
    VkDispatchErr(DispatchError),
    VkCmdBufBuildErr(BuildError),
    VkExecErr(CommandBufferExecError),
}

use JuliaImageError::*;

impl Display for JuliaImageError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            VkImageErr(e) => write!(f, "{}", e),
            VkAllocErr(e) => write!(f, "{}", e),
            VkDescSetErr(e) => write!(f, "{}", e),
            VkOomErr(e) => write!(f, "{}", e),
            VkDispatchErr(e) => write!(f, "{}", e),
            VkCmdBufBuildErr(e) => write!(f, "{}", e),
            VkExecErr(e) => write!(f, "{}", e),
        }
    }
}

impl Error for JuliaImageError {}

impl From<ImageCreationError> for JuliaImageError {
    fn from(err: ImageCreationError) -> JuliaImageError {
        VkImageErr(err)
    }
}

impl From<DeviceMemoryAllocError> for JuliaImageError {
    fn from(err: DeviceMemoryAllocError) -> JuliaImageError {
        VkAllocErr(err)
    }
}

impl From<PersistentDescriptorSetBuildError> for JuliaImageError {
    fn from(err: PersistentDescriptorSetBuildError) -> JuliaImageError {
        VkDescSetErr(err)
    }
}

impl From<OomError> for JuliaImageError {
    fn from(err: OomError) -> JuliaImageError {
        VkOomErr(err)
    }
}

impl From<DispatchError> for JuliaImageError {
    fn from(err: DispatchError) -> JuliaImageError {
        VkDispatchErr(err)
    }
}

impl From<BuildError> for JuliaImageError {
    fn from(err: BuildError) -> JuliaImageError {
        VkCmdBufBuildErr(err)
    }
}

impl From<CommandBufferExecError> for JuliaImageError {
    fn from(err: CommandBufferExecError) -> JuliaImageError {
        VkExecErr(err)
    }
}

impl Debug for JuliaImage {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("JuliaImage")
            .field("image", &self.image)
            .finish()
    }
}
