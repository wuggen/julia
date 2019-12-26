pub mod julia_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/julia.comp",
        //dump: true
    }
}

pub mod julia_vert {
    use gramit::{Vec2, Vec3};

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/julia.vert",
        //dump: true
    }

    #[derive(Debug, Clone, Default)]
    pub struct Vertex {
        in_pos: [f32; 3],
        in_uv: [f32; 2],
    }

    vulkano::impl_vertex!(Vertex, in_pos, in_uv);

    impl Vertex {
        pub fn new(pos: Vec3, uv: Vec2) -> Vertex {
            Vertex {
                in_pos: [pos.x, pos.y, pos.z],
                in_uv: [uv.x, uv.y],
            }
        }

        pub fn fullscreen_quad() -> [Vertex; 6] {
            [
                Vertex::new(vec3!(1.0, 1.0, 0.0), vec2!(1.0, 1.0)),
                Vertex::new(vec3!(-1.0, 1.0, 0.0), vec2!(0.0, 1.0)),
                Vertex::new(vec3!(1.0, -1.0, 0.0), vec2!(1.0, 0.0)),

                Vertex::new(vec3!(-1.0, -1.0, 0.0), vec2!(0.0, 0.0)),
                Vertex::new(vec3!(1.0, -1.0, 0.0), vec2!(1.0, 0.0)),
                Vertex::new(vec3!(-1.0, 1.0, 0.0), vec2!(0.0, 1.0)),
            ]
        }
    }
}

pub mod julia_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/julia.frag",
        //dump: true
    }
}
