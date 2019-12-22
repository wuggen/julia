use vulkano::device::{Device, DeviceCreationError, DeviceExtensions, Features, Queue};
use vulkano::instance::{
    Instance, InstanceCreationError, InstanceExtensions, PhysicalDevice, PhysicalDeviceType,
    QueueFamily,
};
use vulkano::pipeline::ComputePipelineCreationError;
use vulkano::OomError;

use gramit::{Vec2, Vec4};

use std::fmt::{self, Debug, Display, Formatter};
use std::iter;
use std::path::Path;
use std::sync::Arc;

mod export;
mod shaders;

use export::JuliaExport;
use shaders::julia_comp;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JuliaData {
    pub color: [Vec4; 3],
    pub color_midpoint: f32,

    pub n: u32,
    pub c: Vec2,

    pub iters: u32,

    pub center: Vec2,
    pub extents: Vec2,
}

impl JuliaData {
    fn into_shader_data(self) -> julia_comp::ty::Data {
        let mut color = [[0f32; 4]; 3];
        for (i, arr) in color.iter_mut().enumerate() {
            arr.copy_from_slice(self.color[i].as_ref());
        }

        let mut c = [0f32; 2];
        c.copy_from_slice(self.c.as_ref());

        let mut center = [0f32; 2];
        center.copy_from_slice(self.center.as_ref());

        let mut extents = [0f32; 2];
        extents.copy_from_slice(self.extents.as_ref());

        julia_comp::ty::Data {
            color,
            midpt: self.color_midpoint,
            n: self.n,
            c,
            iters: self.iters,
            _dummy0: [0; 4],
            center,
            extents,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImgDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct JuliaContext {
    vk_data: JuliaVkData,
    export: JuliaExport,
}

impl JuliaContext {
    pub fn new() -> Result<JuliaContext, JuliaCreationError> {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .map_err(JuliaCreationError::InstanceCreation)?;
        let (physical, queue_family) =
            find_best_physical_device(&instance).ok_or(JuliaCreationError::DeviceDiscovery)?;
        let (device, mut queues) = Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions::supported_by_device(physical),
            iter::once((queue_family, 0.5)),
        )
        .map_err(JuliaCreationError::DeviceCreation)?;
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

    pub fn export(&self, dims: ImgDimensions, data: &JuliaData, filename: &Path) {
        self.export
            .export(dims, data, filename, self);
    }
}

#[derive(Debug, Clone)]
struct JuliaVkData {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
        .max_by_key(num_compute_queues);

    if let Some(d) = dev {
        d.queue_families()
            .find(QueueFamily::supports_compute)
            .map(|q| (d, q))
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
