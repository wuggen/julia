use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{
    Instance, InstanceExtensions, PhysicalDevice, PhysicalDeviceType, QueueFamily,
};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

use image::{ImageBuffer, Pixel, Rgba};

#[macro_use]
extern crate gramit;
use gramit::{Vec2, Vec4};

use std::iter;
use std::sync::Arc;
use std::time::Instant;

const IMG_DIMS: [u32; 2] = [1024, 1024];

macro_rules! offset_of {
    ($ty:ty, $memb:ident) => {
        unsafe { (&(*(0 as *const $ty)).$memb) as *const _ as usize }
    };
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct JuliaData {
    color: [Vec4; 3],
    midpt: f32,

    n: u32,
    c: Vec2,

    iters: u32,
}

mod julia_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/julia.comp"
    }
}

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
    find_best_by_type(instance, PhysicalDeviceType::DiscreteGpu).or(find_best_by_type(
        instance,
        PhysicalDeviceType::IntegratedGpu,
    )
    .or(
        find_best_by_type(instance, PhysicalDeviceType::VirtualGpu).or(find_best_by_type(
            instance,
            PhysicalDeviceType::Cpu,
        )
        .or(find_best_by_type(instance, PhysicalDeviceType::Other))),
    ))
}

fn main() {
    assert_eq!(offset_of!(JuliaData, color), 0);
    assert_eq!(offset_of!(JuliaData, midpt), 48);
    assert_eq!(offset_of!(JuliaData, n), 52);
    assert_eq!(offset_of!(JuliaData, c), 56);
    assert_eq!(offset_of!(JuliaData, iters), 64);

    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create Vulkan instance");
    let (physical, queue_family) =
        find_best_physical_device(&instance).expect("failed to find a compute-enabled device");

    println!("Using device {} ({:?})", physical.name(), physical.ty());

    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions::supported_by_device(physical),
        iter::once((queue_family, 0.5)),
    )
    .expect("failed to create Vulkan context");
    let queue = queues.next().unwrap();

    let data = JuliaData {
        color: [
            vec4!(0.0, 0.0, 0.0, 1.0),
            vec4!(0.1, 0.5, 0.2, 1.0),
            vec4!(1.0, 1.0, 1.0, 1.0),
        ],
        midpt: 0.25,
        n: 2,
        c: vec2!(-0.750, 0.002),

        iters: 100,
    };

    let (buf, future) =
        ImmutableBuffer::from_data(data, BufferUsage::all(), queue.clone()).unwrap();
    future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: IMG_DIMS[0],
            height: IMG_DIMS[1],
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
    .unwrap();

    let shader = julia_cs::Shader::load(device.clone()).expect("failed to create shader module");
    let compute_pipeline =
        Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap());
    let desc_set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_image(image.clone())
            .unwrap()
            .add_buffer(buf.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    let out_buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..IMG_DIMS[0] * IMG_DIMS[1] * 4).map(|_| 0u8),
    )
    .unwrap();

    let cmd_buf = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [IMG_DIMS[0] / 8, IMG_DIMS[1] / 8, 1],
            compute_pipeline.clone(),
            desc_set.clone(),
            (),
        )
        .unwrap()
        .build()
        .unwrap();

    let cmd_buf2 = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_image_to_buffer(image.clone(), out_buf.clone())
        .unwrap()
        .build()
        .unwrap();

    let now = Instant::now();
    cmd_buf
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let elapsed = now.elapsed();
    println!("Compute time: {:?}", elapsed);

    cmd_buf2
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let img_contents = out_buf.read().unwrap();
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(IMG_DIMS[0], IMG_DIMS[1], &img_contents[..]).unwrap();

    for px in image.pixels() {
        assert_eq!(px.channels()[3], 255);
    }

    image.save("julia.png").unwrap();
}
