use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::device::{Device, DeviceCreationError, DeviceExtensions, Features, Queue};
use vulkano::instance::{
    Instance, InstanceCreationError, InstanceExtensions, PhysicalDevice, PhysicalDeviceType,
    QueueFamily,
};
use vulkano::pipeline::{ComputePipeline, ComputePipelineCreationError};
use vulkano::OomError;

use palette::Srgba;

#[macro_use]
extern crate gramit;
use gramit::{Vec2, Vec4};

use std::fmt::{self, Debug, Display, Formatter};
use std::iter;
use std::path::Path;
use std::sync::Arc;

macro_rules! impl_error {
    (pub enum $enum_name:ident { $($enum_var:ident ($base_err:ty)),* ,}) => {
        impl_error! {
            pub enum $enum_name { $($enum_var($base_err)),* }
        }
    };

    (pub enum $enum_name:ident { $($enum_var:ident ($base_err:ty)),*}) => {
        #[derive(Debug)]
        pub enum $enum_name {
            $($enum_var($base_err)),*
        }

        use $enum_name::*;

        impl Display for $enum_name {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                match self {
                    $($enum_var(e) => write!(f, "{}: {}", stringify!($enum_name), e)),*
                }
            }
        }

        impl Error for $enum_name {}

        $(impl From<$base_err> for $enum_name {
            #[inline]
            fn from(err: $base_err) -> $enum_name {
                $enum_var(err)
            }
        })*
    };
}

mod export;
mod image;
mod render;
mod shaders;

pub mod interface;

use export::JuliaExport;
use shaders::julia_comp;

type CompDesc = PipelineLayout<julia_comp::Layout>;

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
        for (c, orig) in color.iter_mut().zip(self.color.iter()) {
            let nonlin = Srgba::new(orig.x, orig.y, orig.z, orig.w);
            let lin = nonlin.into_linear();
            let (r, g, b, a) = lin.into_components();
            *c = [r, g, b, a];
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
        let instance = Instance::new(
            None,
            &InstanceExtensions {
                ext_debug_utils: true,
                ..vulkano_win::required_extensions()
            },
            None,
        )
        .map_err(JuliaCreationError::InstanceCreation)?;

        let _dbcallback = vulkano::instance::debug::DebugCallback::new(
            &instance,
            vulkano::instance::debug::MessageSeverity {
                error: true,
                warning: true,
                information: false,
                verbose: false,
            },
            vulkano::instance::debug::MessageType {
                general: true,
                validation: true,
                performance: true,
            },
            |m| {
                eprintln!("[{}] {}", m.layer_prefix, m.description);
            },
        )
        .expect("failed to register debug callback");

        let (physical, queue_family) =
            find_best_physical_device(&instance).ok_or(JuliaCreationError::DeviceDiscovery)?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let (device, mut queues) = Device::new(
            physical,
            &Features::none(),
            &device_extensions,
            iter::once((queue_family, 0.5)),
        )
        .map_err(JuliaCreationError::DeviceCreation)?;
        let queue = queues.next().unwrap();

        let shader =
            julia_comp::Shader::load(device.clone()).map_err(JuliaCreationError::ShaderLoad)?;
        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
                .map_err(JuliaCreationError::ComputePipelineCreation)?,
        );

        let export = JuliaExport::new();

        let vk_data = JuliaVkData {
            instance,
            device,
            queue,
            pipeline,
        };

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

    pub fn pipeline(&self) -> &Arc<ComputePipeline<CompDesc>> {
        &self.vk_data.pipeline
    }

    pub fn export(&self, dims: ImgDimensions, data: &JuliaData, filename: &Path) {
        self.export.export(dims, data, filename, self);
    }
}

#[derive(Debug, Clone)]
struct JuliaVkData {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline<CompDesc>>,
}

#[derive(Debug)]
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
