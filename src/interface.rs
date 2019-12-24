use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::pipeline::ComputePipeline;
use vulkano::swapchain::{self, CompositeAlpha, PresentMode, Surface, Swapchain};
use vulkano::sync::{GpuFuture, SharingMode};

use vulkano_win::VkSurfaceBuild;

use winit::dpi::{LogicalPosition, LogicalSize};
use winit::{ElementState, Event, EventsLoop, MouseButton, Window, WindowBuilder, WindowEvent};

use gramit::{Vec2, Vec4, Vector};

use crate::shaders::julia_comp;
use crate::{CompDesc, JuliaContext, JuliaCreationError, JuliaData};

use std::fmt::{self, Debug, Formatter};
use std::sync::{Arc, Mutex};

pub struct JuliaInterface {
    events_loop: EventsLoop,
    state: JuliaState,
    surface: Arc<Surface<Window>>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    descriptor_sets_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<ComputePipeline<CompDesc>>>>,
    buffer_pool: CpuBufferPool<julia_comp::ty::Data>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct JuliaState {
    set_data: JuliaData,
    mouse_state: MouseState,
    close_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MouseState {
    pos: LogicalPosition,
    dragging: bool,
}

impl JuliaState {
    pub fn zoom(&mut self, factor: f32) {
        self.set_data.extents *= factor;
    }

    pub fn pan(&mut self, offset: Vec2) {
        self.set_data.center += offset;
    }

    pub fn set_c(&mut self, c: Vec2) {
        self.set_data.c = c;
    }

    pub fn set_n(&mut self, n: u32) {
        self.set_data.n = n;
    }

    pub fn set_iters(&mut self, iters: u32) {
        self.set_data.iters = iters;
    }

    pub fn close(&mut self) {
        self.close_requested = true;
    }

    pub fn julia_set(&self) -> &JuliaData {
        &self.set_data
    }

    pub fn close_requested(&self) -> bool {
        self.close_requested
    }
}

impl MouseState {
    fn start_drag(&mut self) {
        self.dragging = true;
    }

    fn stop_drag(&mut self) {
        self.dragging = false;
    }

    fn update_and_get_offset(
        &mut self,
        new_pos: LogicalPosition,
        win_size: LogicalSize,
        state: &JuliaData,
    ) -> Vec2 {
        let offset = if self.dragging {
            let diff = vec2!(new_pos.x as f32, new_pos.y as f32)
                - vec2!(self.pos.x as f32, self.pos.y as f32);
            let dims = {
                let (width, height): (f64, f64) = win_size.into();
                vec2!(width as f32, height as f32)
            };
            let ratio = diff / dims;

            ratio * state.extents
        } else {
            vec2!(0.0, 0.0)
        };

        self.pos = new_pos;
        offset
    }
}

impl Debug for JuliaInterface {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("JuliaInterface")
            .field("state", &self.state)
            .field("surface", &self.surface)
            .field("swapchain", &self.swapchain)
            .field("events_loop", &self.events_loop)
            .finish()
    }
}

fn default_state() -> JuliaData {
    JuliaData {
        color: [Vec4::zeros(), Vec4::ones(), Vec4::ones()],
        color_midpoint: 0.25,
        n: 2,
        c: vec2!(0.2, 0.0),
        iters: 100,
        center: Vec2::zeros(),
        extents: vec2!(3.6, 3.6),
    }
}

fn event_callback<'ifc>(
    julia_state: &'ifc mut JuliaState,
    window_dims: LogicalSize,
) -> impl FnMut(Event) + 'ifc {
    move |e| {
        if let Event::WindowEvent { event, .. } = e {
            match event {
                WindowEvent::CursorMoved { position, .. } => {
                    let offset = julia_state.mouse_state.update_and_get_offset(
                        position,
                        window_dims,
                        &julia_state.set_data,
                    );
                    julia_state.pan(offset);
                }

                WindowEvent::MouseInput { state, button, .. } => {
                    if let MouseButton::Left = button {
                        match state {
                            ElementState::Pressed => julia_state.mouse_state.start_drag(),
                            ElementState::Released => julia_state.mouse_state.stop_drag(),
                        }
                    }
                }

                WindowEvent::CloseRequested => julia_state.close(),

                _ => (),
            }
        }
    }
}

impl JuliaInterface {
    pub fn new(
        context: &JuliaContext,
        init_state: Option<JuliaData>,
    ) -> Result<JuliaInterface, JuliaCreationError> {
        let events_loop = EventsLoop::new();

        let monitor = events_loop.get_primary_monitor();

        /*
        let phys_size = monitor.get_dimensions();
        let log_size = phys_size.to_logical(monitor.get_hidpi_factor());

        let min_dim = f64::min(log_size.width, log_size.height);
        let win_dim = min_dim * 0.8;

        let phys_min_dim = f64::min(phys_size.width, phys_size.height);
        let phys_win_dim = phys_min_dim * 0.8;
        */

        //let win_size = LogicalSize::new(win_dim, win_dim);
        let win_size = LogicalSize::new(640.0, 640.0);
        let phys_win_dim = win_size.to_physical(monitor.get_hidpi_factor()).height;

        let surface = WindowBuilder::new()
            .with_dimensions(win_size)
            .with_resizable(false)
            .build_vk_surface(&events_loop, context.instance().clone())
            .map_err(JuliaCreationError::SurfaceCreation)?;

        let caps = surface
            .capabilities(context.device().physical_device())
            .unwrap();
        let image_count = match caps.max_image_count {
            None => u32::max(2, caps.min_image_count),
            Some(lim) => u32::min(u32::max(2, caps.min_image_count), lim),
        };
        let (format, _) = caps.supported_formats[0];
        eprintln!("Supported formats: {:?}", caps.supported_formats);
        eprintln!("Format: {:?}", format);
        let dimensions = caps.current_extent.unwrap_or({
            let w = phys_win_dim.round() as u32;
            [w, w]
        });
        let layers = 1;
        let usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };
        let sharing = SharingMode::Exclusive(context.queue().family().id());
        let transform = caps.current_transform;
        let alpha = CompositeAlpha::Opaque;
        eprintln!("Supported composite alpha: {:?}", caps.supported_composite_alpha);
        let present_mode = PresentMode::Mailbox;
        let clipped = true;
        let old_swapchain = None;

        let (swapchain, swapchain_images) = Swapchain::new(
            context.device().clone(),
            surface.clone(),
            image_count,
            format,
            dimensions,
            layers,
            usage,
            sharing,
            transform,
            alpha,
            present_mode,
            clipped,
            old_swapchain,
        )
        .map_err(JuliaCreationError::SwapchainCreation)?;

        let descriptor_sets_pool = Mutex::new(FixedSizeDescriptorSetsPool::new(
            context.pipeline().clone(),
            0,
        ));
        let buffer_pool = CpuBufferPool::uniform_buffer(context.device().clone());

        Ok(JuliaInterface {
            events_loop,
            state: JuliaState {
                set_data: init_state.unwrap_or_else(default_state),
                mouse_state: MouseState {
                    pos: LogicalPosition { x: 0.0, y: 0.0 },
                    dragging: false,
                },
                close_requested: false,
            },
            surface,
            swapchain,
            swapchain_images,
            descriptor_sets_pool,
            buffer_pool,
        })
    }

    fn command_buffer(
        &self,
        context: &JuliaContext,
        image: Arc<SwapchainImage<Window>>,
    ) -> AutoCommandBuffer {
        let buf = self
            .buffer_pool
            .next(self.state.set_data.into_shader_data())
            .unwrap();
        let [width, height] = image.dimensions();
        let desc_set = Arc::new(
            self.descriptor_sets_pool
                .lock()
                .unwrap()
                .next()
                .add_image(image)
                .unwrap()
                .add_buffer(buf)
                .unwrap()
                .build()
                .unwrap(),
        );

        AutoCommandBufferBuilder::primary_one_time_submit(
            context.device().clone(),
            context.queue().family(),
        )
        .unwrap()
        .dispatch(
            [width / 8, height / 8, 1],
            context.pipeline().clone(),
            desc_set.clone(),
            (),
        )
        .unwrap()
        .build()
        .unwrap()
    }

    fn new_frame(&self, context: &JuliaContext) -> impl GpuFuture {
        let (idx, future) = swapchain::acquire_next_image(self.swapchain.clone(), None).unwrap();
        let image = self.swapchain_images[idx].clone();
        let cmd_buf = self.command_buffer(context, image);

        future
            .then_execute(context.queue().clone(), cmd_buf)
            .unwrap()
            .then_swapchain_present(context.queue().clone(), self.swapchain.clone(), idx)
    }

    fn update(&mut self, context: &JuliaContext) {
        let mut new_state = self.state;
        let window_dims = self.surface.window().get_inner_size().unwrap();
        self.events_loop.poll_events(event_callback(&mut new_state, window_dims));
        self.state = new_state;

        let mut finished = self.new_frame(context)
            .then_signal_fence_and_flush()
            .unwrap();

        finished.wait(None).unwrap();
        finished.cleanup_finished();
    }

    pub fn run(&mut self, context: &JuliaContext) {
        //let mut count = 0;
        while !self.state.close_requested {
            self.update(context);

            /*
            count += 1;
            if count % 60 == 0 {
                eprintln!("{:#?}", self.state);
            }
            */
        }
    }
}
