use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, BeginRenderPassError,
    BuildError, CommandBufferExecError, DrawError, DynamicState,
};
use vulkano::descriptor::descriptor_set::{
    FixedSizeDescriptorSetsPool, PersistentDescriptorSetBuildError,
};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::format::Format;
use vulkano::framebuffer::{
    Framebuffer, FramebufferCreationError, RenderPass, RenderPassCreationError, Subpass,
};
use vulkano::image::ImageViewAccess;
use vulkano::memory::DeviceMemoryAllocError;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineCreationError};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::OomError;

use crate::shaders::{julia_frag, julia_vert};
use crate::JuliaContext;

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

type GraphicsPipelineTy = GraphicsPipeline<
    SingleBufferDefinition<julia_vert::Vertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<RenderPass<pass::Desc>>,
>;

pub struct JuliaRender {
    pipeline: Arc<GraphicsPipelineTy>,
    render_pass: Arc<RenderPass<pass::Desc>>,
    buffer: Arc<ImmutableBuffer<[julia_vert::Vertex]>>,
    descriptor_sets_pool: RefCell<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineTy>>>,
}

impl Debug for JuliaRender {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("JuliaRender")
            .field("pipeline", &self.pipeline)
            .field("render_pass", &self.render_pass)
            .finish()
    }
}

impl JuliaRender {
    pub fn new(
        context: &JuliaContext,
        format: Format,
        samples: u32,
    ) -> Result<JuliaRender, JuliaRenderError> {
        let render_pass = Arc::new(RenderPass::new(
            context.device().clone(),
            pass::Desc::new(format, samples),
        )?);

        let (buffer, future) = ImmutableBuffer::from_iter(
            julia_vert::Vertex::fullscreen_quad().to_vec().into_iter(),
            BufferUsage::vertex_buffer(),
            context.queue().clone(),
        )?;

        let vs = julia_vert::Shader::load(context.device().clone())?;
        let fs = julia_frag::Shader::load(context.device().clone())?;

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<julia_vert::Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .fragment_shader(fs.main_entry_point(), ())
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device().clone())?,
        );

        let descriptor_sets_pool =
            RefCell::new(FixedSizeDescriptorSetsPool::new(pipeline.clone(), 0));

        future.then_signal_fence_and_flush()?.wait(None)?;

        Ok(JuliaRender {
            pipeline,
            render_pass,
            buffer,
            descriptor_sets_pool,
        })
    }

    pub fn draw_after<S, C, F>(
        &self,
        future: F,
        sampled_image: S,
        output_image: C,
        context: &JuliaContext,
    ) -> Result<impl GpuFuture, JuliaRenderError>
    where
        S: ImageViewAccess + Send + Sync + 'static,
        C: ImageViewAccess + Send + Sync + 'static,
        F: GpuFuture,
    {
        //let sampler = Sampler::simple_repeat_linear(context.device().clone());
        let sampler = Sampler::new(
            context.device().clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            1.0,
        )?;

        let desc_set = self
            .descriptor_sets_pool
            .borrow_mut()
            .next()
            .add_sampled_image(sampled_image, sampler)
            .unwrap()
            .build()?;

        let dimensions = {
            let [w, h] = output_image.dimensions().width_height();
            [w as f32, h as f32]
        };

        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(output_image)?
                .build()?,
        );

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions,
                depth_range: -1.0..1.0,
            }]),
            ..DynamicState::none()
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            context.device().clone(),
            context.queue().family(),
        )?
        .begin_render_pass(framebuffer, false, vec![[0.0, 0.0, 0.0, 1.0].into()])?
        .draw(
            self.pipeline.clone(),
            &dynamic_state,
            self.buffer.clone(),
            desc_set,
            (),
        )?
        .end_render_pass()?
        .build()?;

        Ok(future.then_execute(context.queue().clone(), command_buffer)?)
    }
}

//#[derive(Debug, Clone)]
impl_error! {
    pub enum JuliaRenderError {
        VkRenderPassCreationErr(RenderPassCreationError),
        VkDeviceAllocErr(DeviceMemoryAllocError),
        VkOomErr(OomError),
        VkFlushErr(FlushError),
        VkGraphicsPipelineErr(GraphicsPipelineCreationError),
        VkDescSetErr(PersistentDescriptorSetBuildError),
        VkFramebufferErr(FramebufferCreationError),
        VkBeginRenderPassErr(BeginRenderPassError),
        VkDrawErr(DrawError),
        VkCommandBufferContextErr(AutoCommandBufferBuilderContextError),
        VkCommandBufferBuildErr(BuildError),
        VkExecErr(CommandBufferExecError),
        VkSamplerErr(SamplerCreationError),
    }
}

/*
use JuliaRenderError::*;

macro_rules! impl_display {
    ($($enum_variant:ident),*) => {
        impl Display for JuliaRenderError {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                match self {
                    $($enum_variant(e) => write!(f, "error in JuliaRender: {}", e)),*
                }
            }
        }
    };

    ($($enum_variant:ident),* , ) => {
        impl_display!($($enum_variant),*);
    };
}

impl_display!(
    VkRenderPassCreationErr,
    VkDeviceAllocErr,
    VkOomErr,
    VkFlushErr,
    VkGraphicsPipelineErr,
    VkDescSetErr,
    VkFramebufferErr,
    VkBeginRenderPassErr,
    VkDrawErr,
    VkCommandBufferContextErr,
    VkCommandBufferBuildErr,
    VkExecError,
);

impl Error for JuliaRenderError {}

macro_rules! impl_from_err {
    ($from_err:ident, $enum_variant:ident) => {
        impl From<$from_err> for JuliaRenderError {
            fn from(err: $from_err) -> JuliaRenderError {
                $enum_variant(err)
            }
        }
    };
}

impl_from_err!(RenderPassCreationError, VkRenderPassCreationErr);
impl_from_err!(DeviceMemoryAllocError, VkDeviceAllocErr);
impl_from_err!(OomError, VkOomErr);
impl_from_err!(FlushError, VkFlushErr);
impl_from_err!(GraphicsPipelineCreationError, VkGraphicsPipelineErr);
impl_from_err!(PersistentDescriptorSetBuildError, VkDescSetErr);
impl_from_err!(FramebufferCreationError, VkFramebufferErr);
impl_from_err!(BeginRenderPassError, VkBeginRenderPassErr);
impl_from_err!(DrawError, VkDrawErr);
impl_from_err!(
    AutoCommandBufferBuilderContextError,
    VkCommandBufferContextErr
);
impl_from_err!(BuildError, VkCommandBufferBuildErr);
impl_from_err!(CommandBufferExecError, VkExecError);
*/

mod pass {
    use vulkano::format::{ClearValue, Format};
    use vulkano::framebuffer::{
        AttachmentDescription, LoadOp, PassDependencyDescription, PassDescription, RenderPassDesc,
        RenderPassDescClearValues, StoreOp,
    };
    use vulkano::image::ImageLayout;

    #[derive(Debug, Clone, Copy, PartialEq, Hash)]
    pub struct Desc {
        format: Format,
        samples: u32,
    }

    impl Desc {
        pub fn new(format: Format, samples: u32) -> Desc {
            Desc { format, samples }
        }
    }

    unsafe impl RenderPassDesc for Desc {
        #[inline]
        fn num_attachments(&self) -> usize {
            1
        }

        #[inline]
        fn attachment_desc(&self, id: usize) -> Option<AttachmentDescription> {
            if id == 0 {
                Some(AttachmentDescription {
                    format: self.format,
                    samples: self.samples,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    stencil_load: LoadOp::DontCare,
                    stencil_store: StoreOp::DontCare,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                })
            } else {
                None
            }
        }

        #[inline]
        fn num_subpasses(&self) -> usize {
            1
        }

        #[inline]
        fn subpass_desc(&self, id: usize) -> Option<PassDescription> {
            if id == 0 {
                Some(PassDescription {
                    color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
                    depth_stencil: None,
                    input_attachments: vec![],
                    resolve_attachments: vec![],
                    preserve_attachments: vec![],
                })
            } else {
                None
            }
        }

        #[inline]
        fn num_dependencies(&self) -> usize {
            0
        }

        #[inline]
        fn dependency_desc(&self, _: usize) -> Option<PassDependencyDescription> {
            None
        }
    }

    unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for Desc {
        fn convert_clear_values(&self, v: Vec<ClearValue>) -> Box<dyn Iterator<Item = ClearValue>> {
            Box::new(v.into_iter())
        }
    }
}
