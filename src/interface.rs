use vulkano::image::swapchain::SwapchainImage;
use vulkano::image::ImageUsage;
use vulkano::swapchain::{
    self, AcquireError, CompositeAlpha, PresentMode, Surface, Swapchain, SwapchainCreationError,
};
use vulkano::sync::{GpuFuture, SharingMode};

use vulkano_win::VkSurfaceBuild;

use winit::dpi::{LogicalPosition, LogicalSize};
use winit::{
    ElementState, Event, EventsLoop, KeyboardInput, ModifiersState, MouseButton, MouseScrollDelta,
    VirtualKeyCode, Window, WindowBuilder, WindowEvent,
};

use gramit::{Angle, Vec2, Vec4, Vector};

use palette::{Hsv, RgbHue, Srgb};

use crate::export::{ImgDimensions, JuliaExport};
use crate::image::{JuliaImage, JuliaImageError};
use crate::render::{JuliaRender, JuliaRenderError};
use crate::{JuliaContext, JuliaData};

use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct JuliaInterface {
    events_loop: EventsLoop,
    state: JuliaState,
    surface: Arc<Surface<Window>>,
    swapchain: Mutex<Arc<Swapchain<Window>>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    image: JuliaImage,
    render: JuliaRender,
    export: JuliaExport,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct JuliaState {
    data: JuliaData,
    mouse_state: MouseState,
    hsv_colors: [Hsv; 3],
    active_color: u8,
    active_midpt: u8,
    close_requested: bool,
    export_dimensions: ImgDimensions,
    export_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MouseState {
    pos: LogicalPosition,
    dragging: bool,
}

impl JuliaState {
    pub fn zoom(&mut self, factor: f32) {
        self.data.extents *= factor;
    }

    pub fn set_extents(&mut self, extents: Vec2) {
        self.data.extents = extents;
    }

    pub fn pan(&mut self, offset: Vec2) {
        self.data.center += offset;
    }

    pub fn set_center(&mut self, center: Vec2) {
        self.data.center = center;
    }

    pub fn set_c(&mut self, c: Vec2) {
        self.data.c = c;
    }

    pub fn set_n(&mut self, n: u32) {
        self.data.n = n;
    }

    pub fn set_iters(&mut self, iters: u32) {
        self.data.iters = iters;
    }

    pub fn close(&mut self) {
        self.close_requested = true;
    }

    pub fn extents(&self) -> Vec2 {
        self.data.extents
    }

    pub fn center(&self) -> Vec2 {
        self.data.center
    }

    pub fn c(&self) -> Vec2 {
        self.data.c
    }

    pub fn n(&self) -> u32 {
        self.data.n
    }

    pub fn iters(&self) -> u32 {
        self.data.iters
    }

    pub fn active_color(&self) -> Vec4 {
        let hsv = self.hsv_colors[self.active_color_idx()];
        let rgb = Srgb::from(hsv);
        let (r, g, b) = rgb.into_components();
        vec4!(r, g, b, 1.0)
    }

    pub fn active_color_idx(&self) -> usize {
        self.active_color as usize
    }

    pub fn set_active_color(&mut self, idx: usize) -> Result<(), &'static str> {
        if idx < 3 {
            self.active_color = idx as u8;
            Ok(())
        } else {
            Err("index out of range")
        }
    }

    pub fn active_midpt(&self) -> f32 {
        self.data.color_midpoint[self.active_midpt_idx()]
    }

    pub fn active_midpt_idx(&self) -> usize {
        self.active_midpt as usize
    }

    pub fn set_active_midpt(&mut self, idx: usize) -> Result<(), &'static str> {
        if idx < 3 {
            self.active_midpt = idx as u8;
            Ok(())
        } else {
            Err("index out of range")
        }
    }

    pub fn adjust_active_midpt(&mut self, amount: f32) {
        let idx = self.active_midpt_idx();

        let midpt = self.data.color_midpoint[idx] + amount;
        for (i, m) in self.data.color_midpoint.iter_mut().enumerate() {
            if i < idx && *m > midpt {
                *m = midpt;
            } else if i > idx && *m < midpt {
                *m = midpt;
            }
        }

        self.data.color_midpoint[idx] = midpt;
    }

    pub fn adjust_hue(&mut self, amount: f32) {
        let mut hsv = self.hsv_colors[self.active_color_idx()];
        let mut hue = Angle::from_radians(hsv.hue.to_radians());
        hue += Angle::from_degrees(amount);
        hsv.hue = RgbHue::from_radians(hue.radians());
        self.hsv_colors[self.active_color_idx()] = hsv;
        self.data.color[self.active_color_idx()] = self.active_color();
    }

    pub fn adjust_saturation(&mut self, amount: f32) {
        let mut hsv = self.hsv_colors[self.active_color_idx()];
        hsv.saturation += amount / 360.0;
        if hsv.saturation > 1.0 {
            hsv.saturation = 1.0;
        } else if hsv.saturation < 0.0 {
            hsv.saturation = 0.0;
        }
        self.hsv_colors[self.active_color_idx()] = hsv;
        self.data.color[self.active_color_idx()] = self.active_color();
    }

    pub fn adjust_value(&mut self, amount: f32) {
        let mut hsv = self.hsv_colors[self.active_color_idx()];
        hsv.value += amount / 360.0;
        if hsv.value > 1.0 {
            hsv.value = 1.0;
        } else if hsv.value < 0.0 {
            hsv.value = 0.0;
        }
        self.hsv_colors[self.active_color_idx()] = hsv;
        self.data.color[self.active_color_idx()] = self.active_color();
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
        vec2!(-offset.x, offset.y)
    }
}

impl Debug for JuliaInterface {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("JuliaInterface")
            .field("state", &self.state)
            .field("surface", &self.surface)
            .field("swapchain", &self.swapchain)
            .field("events_loop", &self.events_loop)
            .field("image", &self.image)
            .field("render", &self.render)
            .finish()
    }
}

fn default_state() -> JuliaData {
    JuliaData {
        color: [Vec4::zeros(), Vec4::ones(), Vec4::ones()],
        color_midpoint: [0.0, 0.25, 1.0],
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
                        &julia_state.data,
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

                WindowEvent::MouseWheel { delta, .. } => {
                    let factor: f32 = match delta {
                        MouseScrollDelta::LineDelta(_, y) => 0.5 * y as f32,
                        MouseScrollDelta::PixelDelta(LogicalPosition { y, .. }) => 0.5 * y as f32,
                    };

                    let factor = if factor < 0.0 {
                        1.0 + factor.abs()
                    } else {
                        1.0 / (1.0 + factor)
                    };

                    julia_state.zoom(factor)
                }

                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            modifiers,
                            ..
                        },
                    ..
                } => {
                    if let ElementState::Pressed = state {
                        if let Some(code) = virtual_keycode {
                            match code {
                                VirtualKeyCode::Q | VirtualKeyCode::Escape => julia_state.close(),
                                VirtualKeyCode::Add | VirtualKeyCode::Equals => {
                                    julia_state.zoom(1.0 / 1.1)
                                }
                                VirtualKeyCode::Subtract => julia_state.zoom(1.1),
                                VirtualKeyCode::Up
                                | VirtualKeyCode::Down
                                | VirtualKeyCode::Left
                                | VirtualKeyCode::Right => move_c(julia_state, code, modifiers),
                                VirtualKeyCode::PageUp => julia_state.set_n(julia_state.n() + 1),
                                VirtualKeyCode::PageDown => {
                                    let n = julia_state.n();
                                    if n > 2 {
                                        julia_state.set_n(julia_state.n() - 1);
                                    }
                                }

                                VirtualKeyCode::C => julia_state.set_center(vec2!(0.0, 0.0)),
                                VirtualKeyCode::Z => julia_state.set_extents(vec2!(3.6, 3.6)),

                                VirtualKeyCode::RBracket => {
                                    julia_state.set_iters(julia_state.iters() + 10)
                                }

                                VirtualKeyCode::LBracket => {
                                    let iters = julia_state.iters();
                                    if iters <= 20 {
                                        julia_state.set_iters(10);
                                    } else {
                                        julia_state.set_iters(iters - 10);
                                    }
                                }

                                VirtualKeyCode::Key1 => julia_state.set_active_color(0).unwrap(),
                                VirtualKeyCode::Key2 => julia_state.set_active_color(1).unwrap(),
                                VirtualKeyCode::Key3 => julia_state.set_active_color(2).unwrap(),

                                VirtualKeyCode::Key4 => julia_state.set_active_midpt(0).unwrap(),
                                VirtualKeyCode::Key5 => julia_state.set_active_midpt(1).unwrap(),
                                VirtualKeyCode::Key6 => julia_state.set_active_midpt(2).unwrap(),

                                VirtualKeyCode::R => julia_state.adjust_hue(5.0),
                                VirtualKeyCode::F => julia_state.adjust_hue(-5.0),
                                VirtualKeyCode::T => julia_state.adjust_saturation(5.0),
                                VirtualKeyCode::G => julia_state.adjust_saturation(-5.0),
                                VirtualKeyCode::Y => julia_state.adjust_value(2.5),
                                VirtualKeyCode::H => julia_state.adjust_value(-2.5),
                                VirtualKeyCode::U => julia_state.adjust_active_midpt(0.01),
                                VirtualKeyCode::J => julia_state.adjust_active_midpt(-0.01),
                                VirtualKeyCode::I => julia_state.export_dimensions.width += 40,
                                VirtualKeyCode::K => {
                                    if julia_state.export_dimensions.width > 40 {
                                        julia_state.export_dimensions.width -= 40;
                                    }
                                }
                                VirtualKeyCode::O => julia_state.export_dimensions.height += 40,
                                VirtualKeyCode::L => {
                                    if julia_state.export_dimensions.height > 40 {
                                        julia_state.export_dimensions.height -= 40;
                                    }
                                }

                                VirtualKeyCode::W => {
                                    julia_state.pan(vec2!(0.0, julia_state.extents().y / 30.0))
                                }
                                VirtualKeyCode::A => {
                                    julia_state.pan(vec2!(-julia_state.extents().y / 30.0, 0.0))
                                }
                                VirtualKeyCode::S => {
                                    julia_state.pan(vec2!(0.0, -julia_state.extents().y / 30.0))
                                }
                                VirtualKeyCode::D => {
                                    julia_state.pan(vec2!(julia_state.extents().y / 30.0, 0.0))
                                }

                                VirtualKeyCode::E => julia_state.export_requested = true,

                                _ => (),
                            }
                        }
                    }
                }

                WindowEvent::CloseRequested => julia_state.close(),

                _ => (),
            }
        }
    }
}

fn move_c(julia_state: &mut JuliaState, key: VirtualKeyCode, mods: ModifiersState) {
    let dist = 0.001 * {
        if mods.shift {
            0.1
        } else if mods.alt {
            10.0
        } else if mods.ctrl {
            100.0
        } else {
            1.0
        }
    };

    let mut c = julia_state.c();

    match key {
        VirtualKeyCode::Up => c += vec2!(0.0, dist),
        VirtualKeyCode::Down => c -= vec2!(0.0, dist),
        VirtualKeyCode::Left => c -= vec2!(dist, 0.0),
        VirtualKeyCode::Right => c += vec2!(dist, 0.0),
        _ => (),
    }

    julia_state.set_c(c);
}

fn print_state<W: Write>(state: &JuliaState, writer: &mut W) -> io::Result<()> {
    fn fmt_complex(z: Vec2) -> String {
        let op = if z.y < 0.0 { '-' } else { '+' };

        format!("{} {} {}i", z.x, op, z.y.abs())
    }

    fn to_hex(c: Vec4) -> String {
        let c = Srgb::new(c[0], c[1], c[2]);
        let c = Srgb::<u8>::from_format(c);
        format!("{:02x}{:02x}{:02x}", c.red, c.green, c.blue)
    }

    fn wrap_active(s: &str, active: usize, i: usize) -> String {
        let brackets = if i == active { ("[", "]") } else { ("", "") };

        format!("{}{}{}", brackets.0, s, brackets.1)
    }

    fn fmt_color(c: Vec4, active: usize, i: usize) -> String {
        wrap_active(&format!("#{}", to_hex(c)), active, i)
    }

    fn fmt_gradient(
        colors: &[Vec4; 3],
        midpts: &[f32; 3],
        active_color: usize,
        active_midpt: usize,
    ) -> String {
        format!(
            "{}, {}, {} ({}, {}, {})",
            fmt_color(colors[0], active_color, 0),
            fmt_color(colors[1], active_color, 1),
            fmt_color(colors[2], active_color, 2),
            wrap_active(&format!("{}", midpts[0]), active_midpt, 0),
            wrap_active(&format!("{}", midpts[1]), active_midpt, 1),
            wrap_active(&format!("{}", midpts[2]), active_midpt, 2),
        )
    }

    fn fmt_single_hsv(c: Hsv, active: usize, i: usize) -> String {
        wrap_active(
            &format!(
                "H{:5.1} S{:.2} V{:.2}",
                c.hue.to_positive_degrees(),
                c.saturation,
                c.value,
            ),
            active,
            i,
        )
    }

    fn fmt_hsv(colors: &[Hsv; 3], active: usize) -> String {
        format!(
            "{}, {}, {}",
            fmt_single_hsv(colors[0], active, 0),
            fmt_single_hsv(colors[1], active, 1),
            fmt_single_hsv(colors[2], active, 2),
        )
    }

    let range1 = state.center() - 0.5 * state.extents();
    let range2 = state.center() + 0.5 * state.extents();

    writeln!(
        writer,
        r#"
=============================
======= Current state =======
=============================
f(x) = x^{} + ({})
{} Iterations
Range: ({}) -- ({})
Color gradient: {}
    {}
Export dimensions: {}x{}"#,
        state.n(),
        fmt_complex(state.c()),
        state.iters(),
        fmt_complex(range1),
        fmt_complex(range2),
        fmt_gradient(
            &state.data.color,
            &state.data.color_midpoint,
            state.active_color_idx(),
            state.active_midpt_idx(),
        ),
        fmt_hsv(&state.hsv_colors, state.active_color_idx()),
        state.export_dimensions.width,
        state.export_dimensions.height,
    )
}

impl JuliaInterface {
    pub fn new(
        context: &JuliaContext,
        init_state: Option<JuliaData>,
        init_export_dimensions: Option<ImgDimensions>,
    ) -> Result<JuliaInterface, JuliaInterfaceError> {
        let events_loop = EventsLoop::new();

        let monitor = events_loop.get_primary_monitor();

        let phys_size = monitor.get_dimensions();
        let log_size = phys_size.to_logical(monitor.get_hidpi_factor());

        let min_dim = f64::min(log_size.width, log_size.height);
        let win_dim = min_dim * 0.8;

        let phys_min_dim = f64::min(phys_size.width, phys_size.height);
        let phys_win_dim = phys_min_dim * 0.8;

        let win_size = LogicalSize::new(win_dim, win_dim);

        let surface = WindowBuilder::new()
            .with_dimensions(win_size)
            .with_resizable(false)
            .build_vk_surface(&events_loop, context.instance().clone())?;

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
        eprintln!(
            "Supported composite alpha: {:?}",
            caps.supported_composite_alpha
        );
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
        )?;

        let swapchain = Mutex::new(swapchain);

        let image = JuliaImage::new(context, dimensions.clone())?;
        let render = JuliaRender::new(context, format, 1)?;

        let data = init_state.unwrap_or_else(default_state);
        let mut hsv_colors = [Hsv::new(0.0, 0.0, 0.0); 3];
        hsv_colors
            .iter_mut()
            .zip(data.color.iter())
            .for_each(|(hsv, vec)| {
                let rgb = Srgb::new(vec.x, vec.y, vec.z);
                *hsv = Hsv::from(rgb);
            });

        let export_dimensions = init_export_dimensions.unwrap_or_else(|| {
            let [width, height] = dimensions;
            ImgDimensions { width, height }
        });
        let export = JuliaExport::new();

        Ok(JuliaInterface {
            events_loop,
            state: JuliaState {
                data,
                mouse_state: MouseState {
                    pos: LogicalPosition { x: 0.0, y: 0.0 },
                    dragging: false,
                },
                active_color: 0,
                active_midpt: 1,
                hsv_colors,
                close_requested: false,
                export_dimensions,
                export_requested: false,
            },
            surface,
            swapchain,
            swapchain_images,
            image,
            render,
            export,
        })
    }

    fn new_frame(&self, context: &JuliaContext) -> Result<impl GpuFuture, JuliaInterfaceError> {
        let compute_future = self.image.draw(self.state.data, context)?;

        let (idx, acquire_future) =
            swapchain::acquire_next_image(self.swapchain.lock().unwrap().clone(), None)?;
        let swapchain_image = self.swapchain_images[idx].clone();

        Ok(self
            .render
            .draw_after(
                compute_future.join(acquire_future),
                self.image.image().clone(),
                swapchain_image,
                context,
            )?
            .then_swapchain_present(
                context.queue().clone(),
                self.swapchain.lock().unwrap().clone(),
                idx,
            ))
    }

    fn update(&mut self, context: &JuliaContext) -> Result<(), JuliaInterfaceError> {
        let mut new_state = self.state;
        let window_dims = self.surface.window().get_inner_size().unwrap();
        self.events_loop
            .poll_events(event_callback(&mut new_state, window_dims));
        self.state = new_state;

        let mut finished = self
            .new_frame(context)?
            .then_signal_fence_and_flush()
            .unwrap();

        finished.wait(None).unwrap();
        finished.cleanup_finished();

        Ok(())
    }

    pub fn export(&mut self, context: &JuliaContext) {
        let ImgDimensions { width, height } = self.state.export_dimensions;

        let filename = PathBuf::from(format!(
            "{}_{}x{}.png",
            self.state.data.name(),
            width,
            height,
        ));

        let mut export_data = self.state.data;
        if width != height {
            if width < height {
                let ratio = height as f32 / width as f32;
                export_data.extents.y *= ratio;
            } else {
                let ratio = width as f32 / height as f32;
                export_data.extents.x *= ratio;
            }
        }

        print!("Exporting to {} ...", filename.to_str().unwrap());
        io::stdout().flush().unwrap();
        self.export.export(
            self.state.export_dimensions,
            &export_data,
            &filename,
            context,
        );
        println!(" Done!");

        // Ignore any window events that came in during export
        self.events_loop.poll_events(|_| ());
    }

    pub fn run(&mut self, context: &JuliaContext) -> Result<(), JuliaInterfaceError> {
        let mut presented_state = self.state;
        let mut presented_time = Instant::now();
        print_state(&presented_state, &mut io::stdout()).unwrap();

        while !self.state.close_requested() {
            self.update(context)?;

            if presented_time.elapsed().as_secs_f64() > 0.25
                && (self.state.data != presented_state.data
                    || self.state.active_color != presented_state.active_color
                    || self.state.active_midpt != presented_state.active_midpt
                    || self.state.hsv_colors != presented_state.hsv_colors
                    || self.state.export_dimensions != presented_state.export_dimensions)
            {
                presented_state = self.state;
                presented_time = Instant::now();
                print_state(&presented_state, &mut io::stdout()).unwrap();
            }

            if self.state.export_requested {
                self.export(context);
                self.state.export_requested = false;
            }
        }

        Ok(())
    }
}

impl_error! {
    pub enum JuliaInterfaceError {
        JuliaImageErr(JuliaImageError),
        JuliaRenderErr(JuliaRenderError),
        VkWinCreationErr(vulkano_win::CreationError),
        VkSwapchainCreationErr(SwapchainCreationError),
        VkSwapchainAcquireErr(AcquireError),
    }
}
