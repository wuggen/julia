pub mod julia_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/julia.comp",
        //dump: true
    }
}
