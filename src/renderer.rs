use std::collections::HashMap;
use std::ptr::copy_nonoverlapping;
use std::sync::{Arc, RwLock};

use crate::utils::EguiAshCreateInfo;
use ash::vk::*;
use ash::Device;
use ash::Instance;
use bytemuck::cast_slice;
use egui::epaint::{ImageDelta, Primitive};
use egui::{
    ClippedPrimitive, ImageData, TextureFilter, TextureId, TextureOptions, TextureWrapMode,
};
use log::error;
use winit::window::Window;

#[repr(C)]
#[derive(Debug)]
pub(crate) struct EguiVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
    pub color: [u8; 4],
}

pub(crate) struct Renderer {
    pub(crate) window: Arc<Window>,
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    pub(crate) device: Arc<Device>,
    pub(crate) max_texture_side: usize,

    output_in_linear_colorspace: bool,

    graphics_queue: Queue,

    descriptor_pool: DescriptorPool,
    descriptor_set_layout: DescriptorSetLayout,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    command_pools: Vec<CommandPool>,
    command_buffers: Vec<CommandBuffer>,
    vertex_buffers: Vec<Buffer>,
    vertex_buffer_memories: Vec<DeviceMemory>,
    index_buffers: Vec<Buffer>,
    index_buffer_memories: Vec<DeviceMemory>,
    vertex_staging_buffer: Buffer,
    vertex_staging_buffer_memory: DeviceMemory,
    index_staging_buffer: Buffer,
    index_staging_buffer_memory: DeviceMemory,

    framebuffers: Arc<RwLock<Vec<Framebuffer>>>,

    texture_desc_sets: HashMap<TextureId, DescriptorSet>,
    texture_images: HashMap<TextureId, Image>,
    texture_image_views: HashMap<TextureId, ImageView>,
    texture_images_memory: HashMap<TextureId, DeviceMemory>,
    texture_samplers: HashMap<TextureId, Sampler>,

    clipped_primitives: Arc<RwLock<Vec<ClippedPrimitive>>>,
    pixels_per_point: Arc<RwLock<f32>>,

    render_pass: RenderPass,

    frame: usize,
    max_frames_in_flight: usize,

    render_finished: Vec<Semaphore>,
}

impl Renderer {
    pub(crate) fn new(create_info: EguiAshCreateInfo) -> Self {
        let output_in_linear_colorspace = create_info.format == Format::R8G8B8A8_SRGB
            || create_info.format == Format::B8G8R8A8_SRGB
            || create_info.format == Format::R8G8B8_SRGB
            || create_info.format == Format::B8G8R8_SRGB
            || create_info.format == Format::R8G8_SRGB
            || create_info.format == Format::R8_SRGB;
        let max_texture_side = unsafe {
            create_info
                .instance
                .get_physical_device_properties(create_info.physical_device)
        }
        .limits
        .max_image_dimension2_d as usize;
        let image_count = create_info.framebuffers.read().unwrap().len();
        let max_frames_in_flight = { create_info.framebuffers.read().unwrap().len() - 1 };
        let graphics_queue = unsafe {
            create_info.device.get_device_queue(
                create_info.graphics_family_index,
                create_info.graphics_queue_index,
            )
        };
        let render_pass =
            unsafe { Self::create_render_pass(create_info.device.clone(), create_info.format) };
        let (descriptor_pool, descriptor_set_layout) = unsafe {
            Self::create_descriptor(create_info.device.clone(), create_info.texture_capacity)
        };
        let (pipeline, pipeline_layout) = unsafe {
            Self::create_pipeline(
                create_info.device.clone(),
                create_info.window.clone(),
                descriptor_set_layout,
                render_pass,
            )
        };
        let command_pools = unsafe {
            Self::create_command_pools(
                create_info.device.clone(),
                create_info.graphics_family_index,
                image_count,
            )
        };

        let render_finished =
            unsafe { Self::create_sync_objects(create_info.device.clone(), max_frames_in_flight) };

        let (
            vertex_buffers,
            vertex_buffer_memories,
            vertex_staging_buffer,
            vertex_staging_buffer_memory,
            index_buffers,
            index_buffer_memories,
            index_staging_buffer,
            index_staging_buffer_memory,
        ) = unsafe {
            Self::create_buffers(
                create_info.instance.clone(),
                create_info.physical_device,
                create_info.device.clone(),
                image_count,
                create_info.vertex_capacity,
                create_info.index_capacity,
            )
        };

        let mut renderer = Renderer {
            output_in_linear_colorspace,
            max_texture_side,
            instance: create_info.instance,
            physical_device: create_info.physical_device,
            device: create_info.device,
            graphics_queue,
            window: create_info.window,
            descriptor_pool,
            descriptor_set_layout,
            pipeline,
            pipeline_layout,
            command_pools,
            command_buffers: Vec::new(),
            vertex_buffers,
            vertex_buffer_memories,
            index_buffers,
            index_buffer_memories,
            vertex_staging_buffer,
            vertex_staging_buffer_memory,
            index_staging_buffer,
            index_staging_buffer_memory,
            framebuffers: create_info.framebuffers,
            texture_desc_sets: HashMap::new(),
            texture_images: HashMap::new(),
            texture_image_views: HashMap::new(),
            texture_images_memory: HashMap::new(),
            texture_samplers: HashMap::new(),
            clipped_primitives: Arc::new(RwLock::new(Vec::new())),
            pixels_per_point: create_info.the_pixels_per_point,
            render_pass,
            frame: 0,
            max_frames_in_flight,
            render_finished,
        };

        renderer.create_command_buffers();

        renderer
    }

    unsafe fn create_render_pass(device: Arc<Device>, format: Format) -> RenderPass {
        let color_attachment = AttachmentDescription {
            format,
            samples: SampleCountFlags::TYPE_1,
            load_op: AttachmentLoadOp::LOAD,
            store_op: AttachmentStoreOp::STORE,
            stencil_load_op: AttachmentLoadOp::DONT_CARE,
            stencil_store_op: AttachmentStoreOp::DONT_CARE,
            initial_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = AttachmentReference {
            attachment: 0,
            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = SubpassDescription {
            pipeline_bind_point: PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: [color_attachment_ref].as_ptr(),
            ..Default::default()
        };

        let dependency = SubpassDependency {
            src_subpass: SUBPASS_EXTERNAL,
            src_access_mask: AccessFlags::empty(),
            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_subpass: 0,
            dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        };

        let create_info = RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: [color_attachment].as_ptr(),
            subpass_count: 1,
            p_subpasses: [subpass].as_ptr(),
            dependency_count: 1,
            p_dependencies: [dependency].as_ptr(),
            ..Default::default()
        };

        device
            .create_render_pass(&create_info, None)
            .expect("Failed to create render pass")
    }

    unsafe fn create_pipeline(
        device: Arc<Device>,
        window: Arc<Window>,
        descriptor_set_layout: DescriptorSetLayout,
        render_pass: RenderPass,
    ) -> (Pipeline, PipelineLayout) {
        let push_constant_range = PushConstantRange {
            stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
            size: 3 * size_of::<f32>() as u32,
            offset: 0,
        };

        let pipeline_layout_info = PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: [descriptor_set_layout].as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: [push_constant_range].as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let inner_size = window.inner_size();

        let (width, height) = if inner_size.width > 0 && inner_size.height > 0 {
            (inner_size.width, inner_size.height)
        } else {
            (1920, 1080)
        };

        let vert = include_bytes!("shader/shader_vert.spv");
        let frag = include_bytes!("shader/shader_frag.spv");

        let vert_create_info = ShaderModuleCreateInfo {
            p_code: vert.as_ptr() as *const u32,
            code_size: vert.len(),
            ..Default::default()
        };

        let frag_create_info = ShaderModuleCreateInfo {
            p_code: frag.as_ptr() as *const u32,
            code_size: frag.len(),
            ..Default::default()
        };

        let vert_module = device
            .create_shader_module(&vert_create_info, None)
            .unwrap();

        let frag_module = device
            .create_shader_module(&frag_create_info, None)
            .unwrap();

        let vert_stage = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..Default::default()
        };

        let frag_stage = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..Default::default()
        };

        let pipeline_stages = [vert_stage, frag_stage];

        let attributes = {
            let position = VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: 0,
                format: Format::R32G32_SFLOAT,
            };

            let tex_coords = VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                offset: 8,
                format: Format::R32G32_SFLOAT,
            };

            let color = VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                offset: 16,
                format: Format::R8G8B8A8_UNORM,
            };

            [position, tex_coords, color]
        };

        let vertex_binding = VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<EguiVertex>() as u32,
            input_rate: VertexInputRate::VERTEX,
            ..Default::default()
        };

        let vertex_input_info = PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: [vertex_binding].as_ptr(),
            vertex_attribute_description_count: attributes.len() as u32,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly_info = PipelineInputAssemblyStateCreateInfo {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false as u32,
            ..Default::default()
        };

        let viewport_state = Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor_state = Rect2D {
            extent: Extent2D { width, height },
            offset: Offset2D { x: 0, y: 0 },
        };

        let viewport_info = PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            p_viewports: [viewport_state].as_ptr(),
            p_scissors: [scissor_state].as_ptr(),
            ..Default::default()
        };

        let rasterization_info = PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: false as u32,
            rasterizer_discard_enable: false as u32,
            polygon_mode: PolygonMode::FILL,
            cull_mode: CullModeFlags::NONE,
            front_face: FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };

        let stencil_op_state = StencilOpState {
            fail_op: StencilOp::KEEP,
            pass_op: StencilOp::KEEP,
            compare_op: CompareOp::ALWAYS,
            ..Default::default()
        };

        let depth_stencil_info = PipelineDepthStencilStateCreateInfo {
            depth_test_enable: true as u32,
            depth_write_enable: true as u32,
            depth_compare_op: CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: false as u32,
            stencil_test_enable: false as u32,
            front: stencil_op_state,
            back: stencil_op_state,
            ..Default::default()
        };

        let color_blend_attachment_state = PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::RGBA,
            blend_enable: true as u32,
            src_color_blend_factor: BlendFactor::ONE,
            dst_color_blend_factor: BlendFactor::ONE_MINUS_SRC_ALPHA,
            ..Default::default()
        };

        let color_blend_info = PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: [color_blend_attachment_state].as_ptr(),
            ..Default::default()
        };

        let multisample_info = PipelineMultisampleStateCreateInfo {
            rasterization_samples: SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];

        let dynamic_info = PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let graphics_pipeline_info = GraphicsPipelineCreateInfo {
            stage_count: pipeline_stages.len() as u32,
            p_stages: pipeline_stages.as_ptr(),
            p_vertex_input_state: [vertex_input_info].as_ptr(),
            p_input_assembly_state: [input_assembly_info].as_ptr(),
            p_viewport_state: [viewport_info].as_ptr(),
            p_rasterization_state: [rasterization_info].as_ptr(),
            p_multisample_state: [multisample_info].as_ptr(),
            p_depth_stencil_state: [depth_stencil_info].as_ptr(),
            p_color_blend_state: [color_blend_info].as_ptr(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            p_dynamic_state: [dynamic_info].as_ptr(),
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(PipelineCache::null(), &[graphics_pipeline_info], None)
                .expect("Failed to create graphics pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        (graphics_pipeline, pipeline_layout)
    }

    unsafe fn create_descriptor(
        device: Arc<Device>,
        texture_count: usize,
    ) -> (DescriptorPool, DescriptorSetLayout) {
        let descriptor_pool_size = DescriptorPoolSize {
            descriptor_count: texture_count as u32,
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
        };

        let descriptor_pool_info = DescriptorPoolCreateInfo {
            flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            pool_size_count: 1,
            p_pool_sizes: [descriptor_pool_size].as_ptr(),
            max_sets: texture_count as u32,
            ..Default::default()
        };

        let descriptor_set_layout_binding = DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_count: 1,
            descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage_flags: ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };

        let descriptor_set_layout_info = DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: [descriptor_set_layout_binding].as_ptr(),
            ..Default::default()
        };

        let (descriptor_pool, descriptor_set_layout) = {
            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .expect("Failed to create descriptor pool");
            let descriptor_set_layout = device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .expect("Failed to create descriptor set layout");
            (descriptor_pool, descriptor_set_layout)
        };

        (descriptor_pool, descriptor_set_layout)
    }

    unsafe fn create_command_pools(
        device: Arc<Device>,
        queue_family_index: u32,
        image_count: usize,
    ) -> Vec<CommandPool> {
        let pool_info = CommandPoolCreateInfo {
            queue_family_index,
            flags: CommandPoolCreateFlags::TRANSIENT,
            ..Default::default()
        };

        (0..image_count)
            .map(|_| {
                device
                    .create_command_pool(&pool_info, None)
                    .expect("Failed to create command pool")
            })
            .collect()
    }

    unsafe fn create_sync_objects(
        device: Arc<Device>,
        max_frames_in_flight: usize,
    ) -> Vec<Semaphore> {
        let mut render_finished = Vec::with_capacity(max_frames_in_flight);

        let semaphore_info = SemaphoreCreateInfo::default();

        for _ in 0..max_frames_in_flight {
            render_finished.push(device.create_semaphore(&semaphore_info, None).unwrap());
        }

        render_finished
    }

    fn get_type_index(
        memory_properties: &PhysicalDeviceMemoryProperties,
        requirements: MemoryRequirements,
        properties: MemoryPropertyFlags,
    ) -> u32 {
        memory_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|&(i, memory_type)| {
                let suitable = requirements.memory_type_bits & (1 << i) != 0;
                let memory_type = memory_type;

                suitable && memory_type.property_flags.contains(properties)
            })
            .expect("Failed to find suitable memory type")
            .0 as u32
    }

    unsafe fn begin_single_time_commands(
        device: Arc<Device>,
        command_pool: CommandPool,
    ) -> CommandBuffer {
        let command_alloc_buffer_info = CommandBufferAllocateInfo {
            command_pool,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer = device
            .allocate_command_buffers(&command_alloc_buffer_info)
            .unwrap()[0];

        let command_begin_info = CommandBufferBeginInfo {
            flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        device
            .begin_command_buffer(command_buffer, &command_begin_info)
            .unwrap();

        command_buffer
    }

    unsafe fn end_single_time_commands(
        device: Arc<Device>,
        command_pool: CommandPool,
        command_buffer: CommandBuffer,
        graphics_queue: Queue,
    ) {
        device.end_command_buffer(command_buffer).unwrap();

        let submit_info = SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: [command_buffer].as_ptr(),
            ..Default::default()
        };

        device
            .queue_submit(graphics_queue, &[submit_info], Fence::null())
            .unwrap();
        device.queue_wait_idle(graphics_queue).unwrap();

        device.free_command_buffers(command_pool, &[command_buffer]);
    }

    unsafe fn create_buffer(
        instance: Arc<Instance>,
        physical_device: PhysicalDevice,
        device: Arc<Device>,
        size: DeviceSize,
        usage: BufferUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> (Buffer, DeviceMemory) {
        let buffer_info = BufferCreateInfo {
            size,
            usage,
            sharing_mode: SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = device.create_buffer(&buffer_info, None).unwrap();

        let requirements = device.get_buffer_memory_requirements(buffer);

        let memory_properties = instance.get_physical_device_memory_properties(physical_device);

        let type_index = Self::get_type_index(&memory_properties, requirements, properties);

        let memory_info = MemoryAllocateInfo {
            allocation_size: requirements.size,
            memory_type_index: type_index,
            ..Default::default()
        };

        let buffer_memory = device.allocate_memory(&memory_info, None).unwrap();

        device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();

        (buffer, buffer_memory)
    }

    unsafe fn create_image(
        instance: Arc<Instance>,
        physical_device: PhysicalDevice,
        device: Arc<Device>,
        width: u32,
        height: u32,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> (Image, DeviceMemory) {
        let image_info = ImageCreateInfo {
            image_type: ImageType::TYPE_2D,
            extent: Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            format,
            tiling,
            initial_layout: ImageLayout::UNDEFINED,
            usage,
            sharing_mode: SharingMode::EXCLUSIVE,
            samples: SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let image = device.create_image(&image_info, None).unwrap();

        let requirements = device.get_image_memory_requirements(image);

        let memory_properties = instance.get_physical_device_memory_properties(physical_device);

        let type_index = Self::get_type_index(&memory_properties, requirements, properties);

        let memory_info = MemoryAllocateInfo {
            allocation_size: requirements.size,
            memory_type_index: type_index as u32,
            ..Default::default()
        };

        let image_memory = device.allocate_memory(&memory_info, None).unwrap();

        device.bind_image_memory(image, image_memory, 0).unwrap();

        (image, image_memory)
    }

    unsafe fn copy_buffer(
        device: Arc<Device>,
        command_pool: CommandPool,
        graphics_queue: Queue,
        src_buffer: Buffer,
        dst_buffer: Buffer,
        size: DeviceSize,
    ) {
        let command_buffer = Self::begin_single_time_commands(device.clone(), command_pool);

        let copy_region = BufferCopy {
            size,
            ..Default::default()
        };

        device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);

        Self::end_single_time_commands(device, command_pool, command_buffer, graphics_queue);
    }

    unsafe fn copy_buffer_to_image(
        device: Arc<Device>,
        command_pool: CommandPool,
        graphics_queue: Queue,
        src_buffer: Buffer,
        dst_image: Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = Self::begin_single_time_commands(device.clone(), command_pool);

        let subresource = ImageSubresourceLayers {
            aspect_mask: ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let region = BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: subresource,
            image_offset: Offset3D { x: 0, y: 0, z: 0 },
            image_extent: Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        device.cmd_copy_buffer_to_image(
            command_buffer,
            src_buffer,
            dst_image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        Self::end_single_time_commands(device, command_pool, command_buffer, graphics_queue);
    }

    unsafe fn transition_image_layout(
        device: Arc<Device>,
        command_pool: CommandPool,
        graphics_queue: Queue,
        image: Image,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_access_mask: AccessFlags,
        dst_access_mask: AccessFlags,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
    ) {
        let command_buffer = Self::begin_single_time_commands(device.clone(), command_pool);

        let subresource = ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let barrier = ImageMemoryBarrier {
            old_layout,
            new_layout,
            src_queue_family_index: QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: subresource,
            src_access_mask,
            dst_access_mask,
            ..Default::default()
        };

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        Self::end_single_time_commands(device, command_pool, command_buffer, graphics_queue);
    }

    unsafe fn create_buffers(
        instance: Arc<Instance>,
        physical_device: PhysicalDevice,
        device: Arc<Device>,
        count: usize,
        vertex_capacity: usize,
        index_capacity: usize,
    ) -> (
        Vec<Buffer>,
        Vec<DeviceMemory>,
        Buffer,
        DeviceMemory,
        Vec<Buffer>,
        Vec<DeviceMemory>,
        Buffer,
        DeviceMemory,
    ) {
        let vertex_buffer_size = (size_of::<EguiVertex>() * 1024 * vertex_capacity) as DeviceSize;
        let index_buffer_size = (size_of::<u32>() * 1024 * index_capacity) as DeviceSize;

        let (vertex_buffers, vertex_buffer_memories) = (0..count)
            .map(|_| {
                Self::create_buffer(
                    instance.clone(),
                    physical_device,
                    device.clone(),
                    vertex_buffer_size,
                    BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                    MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .unzip();

        let (index_buffers, index_buffer_memories) = (0..count)
            .map(|_| {
                Self::create_buffer(
                    instance.clone(),
                    physical_device,
                    device.clone(),
                    index_buffer_size,
                    BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
                    MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .unzip();

        let (vertex_staging_buffer, vertex_staging_buffer_memory) = Self::create_buffer(
            instance.clone(),
            physical_device,
            device.clone(),
            vertex_buffer_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        let (index_staging_buffer, index_staging_buffer_memory) = Self::create_buffer(
            instance.clone(),
            physical_device,
            device.clone(),
            index_buffer_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        (
            vertex_buffers,
            vertex_buffer_memories,
            vertex_staging_buffer,
            vertex_staging_buffer_memory,
            index_buffers,
            index_buffer_memories,
            index_staging_buffer,
            index_staging_buffer_memory,
        )
    }

    unsafe fn upload_texture(
        &mut self,
        texture_id: TextureId,
        pos: Option<[usize; 2]>,
        [w, h]: [usize; 2],
        options: TextureOptions,
        data: &[u8],
    ) {
        assert_eq!(data.len(), w * h * 4);

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            self.instance.clone(),
            self.physical_device,
            self.device.clone(),
            (w * h * 4) as DeviceSize,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        let memory_ptr = self
            .device
            .map_memory(
                staging_buffer_memory,
                0,
                (w * h * 4) as DeviceSize,
                MemoryMapFlags::empty(),
            )
            .unwrap();

        copy_nonoverlapping(data.as_ptr(), memory_ptr as *mut u8, data.len());

        self.device.unmap_memory(staging_buffer_memory);

        let (image, image_memory) = Self::create_image(
            self.instance.clone(),
            self.physical_device,
            self.device.clone(),
            w as u32,
            h as u32,
            Format::R8G8B8A8_UNORM,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::SAMPLED,
            MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::transition_image_layout(
            self.device.clone(),
            self.command_pools[0],
            self.graphics_queue,
            image,
            ImageLayout::UNDEFINED,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            AccessFlags::empty(),
            AccessFlags::TRANSFER_WRITE,
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
        );

        Self::copy_buffer_to_image(
            self.device.clone(),
            self.command_pools[0],
            self.graphics_queue,
            staging_buffer,
            image,
            w as u32,
            h as u32,
        );

        Self::transition_image_layout(
            self.device.clone(),
            self.command_pools[0],
            self.graphics_queue,
            image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::FRAGMENT_SHADER,
        );

        self.device.destroy_buffer(staging_buffer, None);
        self.device.free_memory(staging_buffer_memory, None);

        let subresource = ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let image_view_info = ImageViewCreateInfo {
            image,
            view_type: ImageViewType::TYPE_2D,
            format: Format::R8G8B8A8_UNORM,
            components: ComponentMapping::default(),
            subresource_range: subresource,
            ..Default::default()
        };

        let image_view = self
            .device
            .create_image_view(&image_view_info, None)
            .unwrap();

        if let Some(pos) = pos {
            if let Some(existing_image) = self.texture_images.get(&texture_id) {
                Self::transition_image_layout(
                    self.device.clone(),
                    self.command_pools[0],
                    self.graphics_queue,
                    *existing_image,
                    ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    AccessFlags::SHADER_READ,
                    AccessFlags::TRANSFER_WRITE,
                    PipelineStageFlags::FRAGMENT_SHADER,
                    PipelineStageFlags::TRANSFER,
                );

                Self::transition_image_layout(
                    self.device.clone(),
                    self.command_pools[0],
                    self.graphics_queue,
                    image,
                    ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ImageLayout::TRANSFER_SRC_OPTIMAL,
                    AccessFlags::SHADER_READ,
                    AccessFlags::TRANSFER_READ,
                    PipelineStageFlags::FRAGMENT_SHADER,
                    PipelineStageFlags::TRANSFER,
                );

                let extent = Offset3D {
                    x: w as i32,
                    y: h as i32,
                    z: 1,
                };

                let top_left = Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                };

                let bottom_right = Offset3D {
                    x: pos[0] as i32 + w as i32,
                    y: pos[1] as i32 + h as i32,
                    z: 1,
                };

                let subresource = ImageSubresourceLayers {
                    aspect_mask: ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                let image_blit = ImageBlit {
                    src_subresource: subresource,
                    src_offsets: [Offset3D { x: 0, y: 0, z: 0 }, extent],
                    dst_subresource: subresource,
                    dst_offsets: [top_left, bottom_right],
                };

                let command_buffer =
                    Self::begin_single_time_commands(self.device.clone(), self.command_pools[0]);

                self.device.cmd_blit_image(
                    command_buffer,
                    image,
                    ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *existing_image,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[image_blit],
                    Filter::NEAREST,
                );

                Self::end_single_time_commands(
                    self.device.clone(),
                    self.command_pools[0],
                    command_buffer,
                    self.graphics_queue,
                );

                Self::transition_image_layout(
                    self.device.clone(),
                    self.command_pools[0],
                    self.graphics_queue,
                    *existing_image,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    AccessFlags::TRANSFER_READ,
                    AccessFlags::SHADER_READ,
                    PipelineStageFlags::TRANSFER,
                    PipelineStageFlags::FRAGMENT_SHADER,
                );

                self.device.destroy_image(image, None);
                self.device.destroy_image_view(image_view, None);
                self.device.free_memory(image_memory, None);
            } else {
                error!("Texture with id {:?} does not exist", texture_id);
                self.device.destroy_image(image, None);
                self.device.destroy_image_view(image_view, None);
                self.device.free_memory(image_memory, None);

                return;
            }
        } else {
            let descriptor_set = {
                let descriptor_set_layouts = [self.descriptor_set_layout];
                let descriptor_set_info = DescriptorSetAllocateInfo {
                    descriptor_pool: self.descriptor_pool,
                    descriptor_set_count: 1,
                    p_set_layouts: descriptor_set_layouts.as_ptr(),
                    ..Default::default()
                };

                let descriptor_sets = self
                    .device
                    .allocate_descriptor_sets(&descriptor_set_info)
                    .unwrap();

                descriptor_sets[0]
            };

            let magnification_filter = match options.magnification {
                TextureFilter::Nearest => Filter::NEAREST,
                TextureFilter::Linear => Filter::LINEAR,
            };

            let minification_filter = match options.minification {
                TextureFilter::Nearest => Filter::NEAREST,
                TextureFilter::Linear => Filter::LINEAR,
            };

            let address_mode = match options.wrap_mode {
                TextureWrapMode::ClampToEdge => SamplerAddressMode::CLAMP_TO_EDGE,
                TextureWrapMode::Repeat => SamplerAddressMode::REPEAT,
                TextureWrapMode::MirroredRepeat => SamplerAddressMode::MIRRORED_REPEAT,
            };

            let sampler_info = SamplerCreateInfo {
                mag_filter: magnification_filter,
                min_filter: minification_filter,
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                address_mode_w: address_mode,
                anisotropy_enable: false as u32,
                mipmap_mode: SamplerMipmapMode::LINEAR,
                min_lod: 0.0,
                max_lod: LOD_CLAMP_NONE,
                ..Default::default()
            };

            let sampler = self.device.create_sampler(&sampler_info, None).unwrap();

            let image_info = DescriptorImageInfo {
                sampler,
                image_view,
                image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let descriptor_write = WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: [image_info].as_ptr(),
                ..Default::default()
            };

            self.device.update_descriptor_sets(&[descriptor_write], &[]);

            if let Some((_, old_descriptor_set)) = self.texture_desc_sets.remove_entry(&texture_id)
            {
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[old_descriptor_set])
                    .expect("Failed to free descriptor set");
            }

            if let Some((_, old_image)) = self.texture_images.remove_entry(&texture_id) {
                self.device.destroy_image(old_image, None);
            }

            if let Some((_, old_image_view)) = self.texture_image_views.remove_entry(&texture_id) {
                self.device.destroy_image_view(old_image_view, None);
            }

            if let Some((_, old_image_memory)) =
                self.texture_images_memory.remove_entry(&texture_id)
            {
                self.device.free_memory(old_image_memory, None);
            }

            if let Some((_, old_sampler)) = self.texture_samplers.remove_entry(&texture_id) {
                self.device.destroy_sampler(old_sampler, None);
            }

            self.texture_desc_sets.insert(texture_id, descriptor_set);
            self.texture_images.insert(texture_id, image);
            self.texture_image_views.insert(texture_id, image_view);
            self.texture_images_memory.insert(texture_id, image_memory);
            self.texture_samplers.insert(texture_id, sampler);
        }
    }

    unsafe fn free_texture(&mut self, texture_id: TextureId) {
        if let Some((_, descriptor_set)) = self.texture_desc_sets.remove_entry(&texture_id) {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                .expect("Failed to free descriptor set");
        }

        if let Some((_, image)) = self.texture_images.remove_entry(&texture_id) {
            self.device.destroy_image(image, None);
        }

        if let Some((_, image_view)) = self.texture_image_views.remove_entry(&texture_id) {
            self.device.destroy_image_view(image_view, None);
        }

        if let Some((_, image_memory)) = self.texture_images_memory.remove_entry(&texture_id) {
            self.device.free_memory(image_memory, None);
        }

        if let Some((_, sampler)) = self.texture_samplers.remove_entry(&texture_id) {
            self.device.destroy_sampler(sampler, None);
        }
    }

    pub(crate) fn update_buffer(&mut self, vertices: &[EguiVertex], indices: &[u32], index: usize) {
        if vertices.is_empty() || indices.is_empty() {
            return;
        }

        unsafe {
            let vertex_size = (vertices.len() * size_of::<EguiVertex>()) as DeviceSize;
            let index_size = (indices.len() * size_of::<u32>()) as DeviceSize;

            let (staging_buffer, staging_buffer_memory) = (
                self.vertex_staging_buffer,
                self.vertex_staging_buffer_memory,
            );

            let memory_ptr = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    vertex_size,
                    MemoryMapFlags::empty(),
                )
                .unwrap();

            copy_nonoverlapping(vertices.as_ptr(), memory_ptr.cast(), vertex_size as usize);

            self.device.unmap_memory(staging_buffer_memory);

            let vertex_buffer = self.vertex_buffers[index];

            Self::copy_buffer(
                self.device.clone(),
                self.command_pools[index],
                self.graphics_queue,
                staging_buffer,
                vertex_buffer,
                vertex_size,
            );

            let (staging_buffer, staging_buffer_memory) =
                (self.index_staging_buffer, self.index_staging_buffer_memory);

            let memory_ptr = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    index_size,
                    MemoryMapFlags::empty(),
                )
                .unwrap();

            copy_nonoverlapping(indices.as_ptr(), memory_ptr.cast(), index_size as usize);

            self.device.unmap_memory(staging_buffer_memory);

            let index_buffer = self.index_buffers[index];

            Self::copy_buffer(
                self.device.clone(),
                self.command_pools[index],
                self.graphics_queue,
                staging_buffer,
                index_buffer,
                index_size,
            );
        }
    }

    pub(crate) fn destroy_buffers(&mut self, index: usize) {
        unsafe {
            if !self.vertex_buffers[index].is_null() {
                self.device.destroy_buffer(self.vertex_buffers[index], None);
            }
            if !self.vertex_buffer_memories[index].is_null() {
                self.device
                    .free_memory(self.vertex_buffer_memories[index], None);
            }
            if !self.index_buffers[index].is_null() {
                self.device.destroy_buffer(self.index_buffers[index], None);
            }
            if !self.index_buffer_memories[index].is_null() {
                self.device
                    .free_memory(self.index_buffer_memories[index], None);
            }
        }
    }

    fn record_command_buffers(&self, extent: Extent2D, index: usize) {
        let command_buffer = self.command_buffers[index];

        let begin_info = CommandBufferBeginInfo {
            flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        let render_pass_info = RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers.read().unwrap()[index],
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent,
            },
            ..Default::default()
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();

            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                SubpassContents::INLINE,
            );
        }

        let clipped_primitives = self.clipped_primitives.read().unwrap();

        unsafe {
            if clipped_primitives.is_empty() {
                self.device.cmd_end_render_pass(command_buffer);
                self.device.end_command_buffer(command_buffer).unwrap();
                return;
            }
        }

        self.command_buffer_core(extent, index);

        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer).unwrap();
        }
    }

    fn command_buffer_core(&self, extent: Extent2D, index: usize) {
        let clipped_primitives = self.clipped_primitives.read().unwrap();
        let command_buffer = self.command_buffers[index];
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffers[index]],
                &[0],
            );

            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffers[index],
                0,
                IndexType::UINT32,
            );

            let viewport = Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);

            let pixels_per_point = *self.pixels_per_point.read().unwrap();
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                0,
                cast_slice(&[extent.width as f32 / pixels_per_point]),
            );

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                4,
                cast_slice(&[extent.height as f32 / pixels_per_point]),
            );

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                8,
                cast_slice(&[self.output_in_linear_colorspace as i32]),
            );

            let mut vertex_offset = 0;
            let mut index_offset = 0;

            for ClippedPrimitive {
                clip_rect,
                primitive,
            } in clipped_primitives.iter()
            {
                match primitive {
                    Primitive::Mesh(mesh) => {
                        if mesh.vertices.is_empty() && mesh.indices.is_empty() {
                            continue;
                        }

                        if let Some(desc_set) = self.texture_desc_sets.get(&mesh.texture_id) {
                            self.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[*desc_set],
                                &[],
                            );
                        }

                        let pixels_per_point = *self.pixels_per_point.read().unwrap();

                        let clip_min_x = clip_rect.min.x * pixels_per_point;
                        let clip_min_y = clip_rect.min.y * pixels_per_point;
                        let clip_max_x = clip_rect.max.x * pixels_per_point;
                        let clip_max_y = clip_rect.max.y * pixels_per_point;

                        let clip_min_x = clip_min_x.round() as u32;
                        let clip_min_y = clip_min_y.round() as u32;
                        let clip_max_x = clip_max_x.round() as u32;
                        let clip_max_y = clip_max_y.round() as u32;

                        let clip_min_x = clip_min_x.clamp(0, extent.width);
                        let clip_min_y = clip_min_y.clamp(0, extent.height);
                        let clip_max_x = clip_max_x.clamp(clip_min_x, extent.width);
                        let clip_max_y = clip_max_y.clamp(clip_min_y, extent.height);

                        let scissor = Rect2D {
                            offset: Offset2D {
                                x: clip_min_x as i32,
                                y: clip_min_y as i32,
                            },
                            extent: Extent2D {
                                width: (clip_max_x - clip_min_x),
                                height: (clip_max_y - clip_min_y),
                            },
                        };

                        self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                        let vertex_count = mesh.vertices.len() as i32;
                        let index_count = mesh.indices.len() as u32;

                        self.device.cmd_draw_indexed(
                            command_buffer,
                            index_count,
                            1,
                            index_offset,
                            vertex_offset,
                            0,
                        );

                        vertex_offset += vertex_count;
                        index_offset += index_count;
                    }
                    Primitive::Callback(_callback) => {
                        unimplemented!()
                    }
                }
            }
        }
    }

    fn create_command_buffers(&mut self) {
        self.command_pools.iter().for_each(|command_pool| {
            let command_buffer_info = CommandBufferAllocateInfo {
                command_pool: *command_pool,
                level: CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };

            let command_buffer =
                unsafe { self.device.allocate_command_buffers(&command_buffer_info) }.unwrap()[0];

            self.command_buffers.push(command_buffer);
        });
    }

    pub(crate) fn register_texture(&mut self, texture_id: TextureId, delta: &ImageDelta) {
        let data: &[u8] = match &delta.image {
            ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );

                &image
                    .pixels
                    .iter()
                    .map(|pixel| pixel.to_array())
                    .flatten()
                    .collect::<Vec<_>>()
            }
            ImageData::Font(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                &image
                    .srgba_pixels(None)
                    .flat_map(|pixel| pixel.to_array())
                    .collect::<Vec<_>>()
            }
        };

        unsafe {
            self.upload_texture(
                texture_id,
                delta.pos,
                [delta.image.width(), delta.image.height()],
                delta.options,
                data,
            );
        }
    }

    pub(crate) fn unregister_texture(&mut self, texture_id: TextureId) {
        unsafe {
            self.free_texture(texture_id);
        }
    }

    pub(crate) fn draw_primitive(
        &mut self,
        clipped_primitives: &[ClippedPrimitive],
        images_available: Semaphore,
        extent: Extent2D,
        image_index: usize,
    ) -> Semaphore {
        let (vertices, indices) = {
            let mut primitives = self.clipped_primitives.write().unwrap();

            primitives.extend_from_slice(clipped_primitives);

            let vertices = primitives
                .iter()
                .filter_map(|clipped_primitive| match &clipped_primitive.primitive {
                    Primitive::Mesh(mesh) => Some(
                        mesh.vertices
                            .iter()
                            .map(|vertex| EguiVertex {
                                position: [vertex.pos.x, vertex.pos.y],
                                tex_coords: [vertex.uv.x, vertex.uv.y],
                                color: vertex.color.to_array(),
                            })
                            .collect::<Vec<_>>(),
                    ),
                    Primitive::Callback(_) => None,
                })
                .flatten()
                .collect::<Vec<_>>();
            let indices = primitives
                .iter()
                .filter_map(|clipped_primitive| match &clipped_primitive.primitive {
                    Primitive::Mesh(mesh) => Some(mesh.indices.iter().copied().collect::<Vec<_>>()),
                    Primitive::Callback(_) => None,
                })
                .flatten()
                .collect::<Vec<_>>();

            (vertices, indices)
        };

        let command_buffer = [self.command_buffers[image_index]];

        let wait_semaphores = [images_available];
        let signal_semaphores = [self.render_finished[self.frame]];
        let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit_info = SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: command_buffer.as_ptr(),
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device
                .reset_command_pool(
                    self.command_pools[image_index],
                    CommandPoolResetFlags::RELEASE_RESOURCES,
                )
                .unwrap();

            self.update_buffer(&vertices, &indices, image_index);

            self.record_command_buffers(extent, image_index);

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], Fence::null())
                .unwrap();

            self.frame = (self.frame + 1) % self.max_frames_in_flight;
        }

        {
            self.clipped_primitives.write().unwrap().clear();
        }

        signal_semaphores[0]
    }

    pub(crate) fn destroy(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            let image_count = { self.framebuffers.read().unwrap().len() };

            for i in 0..image_count {
                self.destroy_buffers(i)
            }
            self.device.destroy_buffer(self.vertex_staging_buffer, None);
            self.device
                .free_memory(self.vertex_staging_buffer_memory, None);
            self.device.destroy_buffer(self.index_staging_buffer, None);
            self.device
                .free_memory(self.index_staging_buffer_memory, None);
            let keys = self.texture_desc_sets.keys().cloned().collect::<Vec<_>>();
            for key in keys {
                self.free_texture(key);
            }
            self.command_pools.iter().enumerate().for_each(|(i, pool)| {
                self.device
                    .free_command_buffers(*pool, &[self.command_buffers[i]]);
                self.device.destroy_command_pool(*pool, None);
            });
            self.render_finished.iter().for_each(|semaphore| {
                self.device.destroy_semaphore(*semaphore, None);
            });
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
