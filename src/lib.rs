use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceCreationError, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{
    Instance, InstanceCreationError, InstanceExtensions, PhysicalDevice, PhysicalDeviceType,
    QueueFamily,
};
use vulkano::pipeline::{ComputePipeline, ComputePipelineCreationError};
use vulkano::sync::GpuFuture;
use vulkano::OomError;

use image::{ImageBuffer, Rgba};

use gramit::{Vec2, Vec4};

use std::cell::Cell;
use std::fmt::{self, Debug, Display, Formatter};
use std::iter;
use std::path::Path;
use std::sync::Arc;

pub mod shaders;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct JuliaData {
    pub color: [Vec4; 3],
    pub color_midpoint: f32,

    pub n: u32,
    pub c: Vec2,

    pub iters: u32,

    pub dimensions: u32,
}

#[derive(Debug)]
pub struct JuliaContext {
    vk_data: JuliaVkData,
    export: JuliaExport,
}

type CompDesc = vulkano::descriptor::pipeline_layout::PipelineLayout<shaders::julia_comp::Layout>;

impl JuliaContext {
    pub fn new() -> Result<JuliaContext, JuliaCreationError> {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .map_err(|e| JuliaCreationError::InstanceCreation(e))?;
        let (physical, queue_family) =
            find_best_physical_device(&instance).ok_or(JuliaCreationError::DeviceDiscovery)?;
        let (device, mut queues) = Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions::supported_by_device(physical),
            iter::once((queue_family, 0.5)),
        )
        .map_err(|e| JuliaCreationError::DeviceCreation(e))?;
        let queue = queues.next().unwrap();

        let vk_data = JuliaVkData {
            instance,
            device,
            queue,
        };

        let export = JuliaExport::new(&vk_data.device)?;

        Ok(JuliaContext { vk_data, export })
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.vk_data.instance
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.vk_data.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.vk_data.queue
    }

    pub fn export(&self, data: &JuliaData, filename: &Path) {
        self.export
            .export(data, filename, self.device(), self.queue());
    }
}

#[derive(Debug, Clone)]
struct JuliaVkData {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

struct JuliaExport {
    pipeline: Arc<ComputePipeline<CompDesc>>,
    cached_data: Cell<Option<JuliaExportCache>>,
}

struct JuliaExportCache {
    data: JuliaData,
    output_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    command_buffer: AutoCommandBuffer<StandardCommandPoolAlloc>,
}

impl JuliaExport {
    fn new(device: &Arc<Device>) -> Result<JuliaExport, JuliaCreationError> {
        let shader = shaders::julia_comp::Shader::load(device.clone())
            .map_err(|e| JuliaCreationError::ShaderLoad(e))?;
        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
                .map_err(|e| JuliaCreationError::ComputePipelineCreation(e))?,
        );

        Ok(JuliaExport {
            pipeline,
            cached_data: Cell::new(None),
        })
    }

    fn regen_cache(&self, data: &JuliaData, device: &Arc<Device>, queue: &Arc<Queue>) {
        let data = data.clone();
        let (input_buffer, future) =
            ImmutableBuffer::from_data(data, BufferUsage::all(), queue.clone()).unwrap();
        let image = StorageImage::new(
            device.clone(),
            Dimensions::Dim2d {
                width: data.dimensions,
                height: data.dimensions,
            },
            Format::R8G8B8A8Unorm,
            Some(queue.family()),
        )
        .unwrap();
        let desc_set = Arc::new(
            PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_image(image.clone())
                .unwrap()
                .add_buffer(input_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );
        let output_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            (0..data.dimensions * data.dimensions * 4).map(|_| 0u8),
        )
        .unwrap();

        let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .dispatch(
                [data.dimensions / 8, data.dimensions / 8, 1],
                self.pipeline.clone(),
                desc_set.clone(),
                (),
            )
            .unwrap()
            .copy_image_to_buffer(image.clone(), output_buffer.clone())
            .unwrap()
            .build()
            .unwrap();

        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.cached_data.set(Some(JuliaExportCache {
            data,
            output_buffer,
            command_buffer,
        }));
    }

    fn export(&self, data: &JuliaData, filename: &Path, device: &Arc<Device>, queue: &Arc<Queue>) {
        let cache_opt = self.cached_data.take();
        if cache_opt.is_none() {
            self.regen_cache(data, device, queue);
        } else {
            let c = cache_opt.unwrap();
            if c.data != *data {
                self.regen_cache(data, device, queue);
            } else {
                self.cached_data.set(Some(c));
            }
        }

        self.export_core(filename, queue);
    }

    fn export_core(&self, filename: &Path, queue: &Arc<Queue>) {
        let cache = self.cached_data.take().unwrap();
        cache
            .command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let img_contents = cache.output_buffer.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            cache.data.dimensions,
            cache.data.dimensions,
            &img_contents[..],
        )
        .unwrap();
        image.save(filename).unwrap();
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

#[derive(Debug, Clone)]
pub enum JuliaCreationError {
    InstanceCreation(InstanceCreationError),
    DeviceDiscovery,
    DeviceCreation(DeviceCreationError),
    ShaderLoad(OomError),
    ComputePipelineCreation(ComputePipelineCreationError),
}

impl Display for JuliaCreationError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            JuliaCreationError::InstanceCreation(e) => write!(f, "{}", e),
            JuliaCreationError::DeviceDiscovery => {
                write!(f, "failed to find a compute-enabled device")
            }
            JuliaCreationError::DeviceCreation(e) => write!(f, "{}", e),
            JuliaCreationError::ShaderLoad(e) => write!(f, "failed to load shader: {}", e),
            JuliaCreationError::ComputePipelineCreation(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for JuliaCreationError {}

fn num_compute_queues(dev: &PhysicalDevice) -> usize {
    let mut total = 0;
    for fam in dev.queue_families() {
        if fam.supports_compute() {
            total += fam.queues_count();
        }
    }

    total
}

fn find_best_by_type(
    instance: &Arc<Instance>,
    ty: PhysicalDeviceType,
) -> Option<(PhysicalDevice, QueueFamily)> {
    let dev = PhysicalDevice::enumerate(instance)
        .filter(|d| {
            d.ty() == ty
                && DeviceExtensions::supported_by_device(*d).khr_storage_buffer_storage_class
        })
        .max_by_key(|d| num_compute_queues(d));

    if let Some(d) = dev {
        d.queue_families()
            .find(|f| f.supports_compute())
            .map(|q| (d, q.clone()))
    } else {
        None
    }
}

fn find_best_physical_device(instance: &Arc<Instance>) -> Option<(PhysicalDevice, QueueFamily)> {
    find_best_by_type(instance, PhysicalDeviceType::DiscreteGpu)
        .or_else(|| find_best_by_type(instance, PhysicalDeviceType::IntegratedGpu))
        .or_else(|| find_best_by_type(instance, PhysicalDeviceType::VirtualGpu))
        .or_else(|| find_best_by_type(instance, PhysicalDeviceType::Cpu))
        .or_else(|| find_best_by_type(instance, PhysicalDeviceType::Other))
}

#[cfg(test)]
mod test {
    use super::*;

    use std::mem;
    use std::ptr;

    macro_rules! offset_of {
        ($ty:ty, $memb:ident) => {
            unsafe { (&(*ptr::null::<$ty>()).$memb) as *const _ as usize }
        };
    }

    #[test]
    fn julia_data_layout() {
        assert_eq!(offset_of!(JuliaData, color), 0);
        assert_eq!(offset_of!(JuliaData, color_midpoint), 48);
        assert_eq!(offset_of!(JuliaData, n), 52);
        assert_eq!(offset_of!(JuliaData, c), 56);
        assert_eq!(offset_of!(JuliaData, iters), 64);
        assert_eq!(offset_of!(JuliaData, dimensions), 68);
        assert_eq!(mem::size_of::<JuliaData>(), 72);
    }
}