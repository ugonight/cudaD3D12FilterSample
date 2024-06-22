#include <windows.h>

#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>

#include <string>
#include <shellapi.h>

#include "DxApplication.h"

#include <stdexcept>
#include <system_error>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "cudart_static.lib")

inline void ThrowIfFailed(HRESULT hr) {
	if (FAILED(hr)) {
		throw std::runtime_error(std::system_category().message(hr));
	}
}

DxApplication::DxApplication(UINT width, UINT height) :
	m_frameIndex(0),
	m_width(width), m_height(height),
	m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
	m_fenceValues{},
	m_rtvDescriptorSize(0),
	m_format(DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT),
	m_clearColor{ 0.1f,0.1f,0.1f,1.0f }
{
	m_viewport = { 0.0f, 0.0f, static_cast<float>(width),
			  static_cast<float>(height) };
	cudaEventCreate(&m_event1);
	cudaEventCreate(&m_event2);
}

void DxApplication::OnInit()
{
	LoadPipeline();
	LoadAssets();
}

void DxApplication::LoadPipeline()
{
	UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
	// Enable the debug layer (requires the Graphics Tools "optional feature").
	// NOTE: Enabling the debug layer after device creation will invalidate the
	// active device.
	{
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
			debugController->EnableDebugLayer();

			// Enable additional debug layers.
			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}
#endif

	ComPtr<IDXGIFactory4> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	{
		for (UINT adapterIndex = 0;
			DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapterIndex, &hardwareAdapter);
			++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			hardwareAdapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
				_uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	}

	ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(),
		D3D_FEATURE_LEVEL_11_0,
		IID_PPV_ARGS(&m_device)));
	DXGI_ADAPTER_DESC1 desc;
	hardwareAdapter->GetDesc1(&desc);
	m_dx12deviceluid = desc.AdapterLuid;

	InitCuda();

	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(
		m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

	// No swap chains are created
#if false
	// Describe and create the swap chain.
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = FrameCount;
	swapChainDesc.Width = m_width;
	swapChainDesc.Height = m_height;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;

	ComPtr<IDXGISwapChain1> swapChain;
	ThrowIfFailed(factory->CreateSwapChainForHwnd(
		m_commandQueue.Get(),  // Swap chain needs the queue so that it can force
		// a flush on it.
		Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
		&swapChain));

	// This sample does not support fullscreen transitions.
	ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(),
		DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain.As(&m_swapChain));
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
#endif

	// Create descriptor heaps.
	{
		// Describe and create a render target view (RTV) descriptor heap.
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = FrameCount;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		ThrowIfFailed(
			m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));
		m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
			D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

		D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
		srvHeapDesc.NumDescriptors = 1;
		srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));
	}

	// Create frame resources.
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
			m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
		auto const heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		const D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(m_format,
			static_cast<UINT64>(m_width),
			static_cast<UINT>(m_height),
			1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

		D3D12_CLEAR_VALUE clearValue = { m_format, {} };
		memcpy(clearValue.Color, m_clearColor, sizeof(clearValue.Color));

		// Create a RTV and a command allocator for each frame.
		for (UINT n = 0; n < FrameCount; n++) {
			//ThrowIfFailed(
			//	m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));

			ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties,
				D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES | D3D12_HEAP_FLAG_SHARED,
				&desc,
				D3D12_RESOURCE_STATE_RENDER_TARGET, &clearValue,
				IID_PPV_ARGS(&m_renderTargets[n])));

			m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr,
				rtvHandle);
			rtvHandle.Offset(1, m_rtvDescriptorSize);

			// Share RenderTarget resources with CUDA
			{
				HANDLE sharedHandle{};

				SECURITY_ATTRIBUTES secAttr{};
				ThrowIfFailed(m_device->CreateSharedHandle(m_renderTargets[n].Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));

				const auto texAllocInfo = m_device->GetResourceAllocationInfo(m_nodeMask, 1, &desc);

				cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
				cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
				cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
				cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
				cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
				cudaImportExternalMemory(&m_externalMemory_out[n], &cuExtmemHandleDesc);
				CloseHandle(sharedHandle);

				cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
				cuExtmemMipDesc.extent = make_cudaExtent(desc.Width, desc.Height, 0);
				cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
				cuExtmemMipDesc.numLevels = 1;
				cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;

				cudaExternalMemoryGetMappedMipmappedArray(&m_cuMipArray_out[n], m_externalMemory_out[n], &cuExtmemMipDesc);
				cudaGetMipmappedArrayLevel(&m_cuArray_out[n], m_cuMipArray_out[n], 0);
			}

			ThrowIfFailed(m_device->CreateCommandAllocator(
				D3D12_COMMAND_LIST_TYPE_DIRECT,
				IID_PPV_ARGS(&m_commandAllocators[n])));
		}
	}
}

void DxApplication::InitCuda()
{
	int num_cuda_devices = 0;
	cudaGetDeviceCount(&num_cuda_devices);

	if (!num_cuda_devices) {
		throw std::exception("No CUDA Devices found");
	}
	for (UINT devId = 0; devId < num_cuda_devices; devId++) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, devId);

		if ((memcmp(&m_dx12deviceluid.LowPart, devProp.luid,
			sizeof(m_dx12deviceluid.LowPart)) == 0) &&
			(memcmp(&m_dx12deviceluid.HighPart,
				devProp.luid + sizeof(m_dx12deviceluid.LowPart),
				sizeof(m_dx12deviceluid.HighPart)) == 0)) {
			cudaSetDevice(devId);
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			cudaStreamCreate(&m_streamToRun);
			printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			break;
		}
	}
}

void DxApplication::LoadAssets()
{
	// Create a root signature.
	{
		CD3DX12_DESCRIPTOR_RANGE1 discriptorRanges[1];
		discriptorRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
		CD3DX12_ROOT_PARAMETER1 rootParameters[1];
		rootParameters[0].InitAsDescriptorTable(1, &discriptorRanges[0], D3D12_SHADER_VISIBILITY_PIXEL);
		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
		sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.MipLODBias = 0;
		sampler.MaxAnisotropy = 0;
		sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
		sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MinLOD = 0.0f;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		sampler.ShaderRegister = 0;
		sampler.RegisterSpace = 0;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
		ComPtr<ID3DBlob> rootSignatureBlob = nullptr;
		ComPtr<ID3DBlob> errorBlob = nullptr;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rootSignatureBlob, &errorBlob));
		ThrowIfFailed(m_device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
	}

	{
#if defined(_DEBUG)
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
		UINT compileFlags = 0;
#endif
		ComPtr<ID3DBlob> vsBlob;
		ComPtr<ID3DBlob> psBlob;
		D3DCompileFromFile(L"Shader.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vsBlob, nullptr);
		D3DCompileFromFile(L"Shader.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &psBlob, nullptr);

		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature = m_rootSignature.Get();
		psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
		psoDesc.VS = CD3DX12_SHADER_BYTECODE(vsBlob.Get());
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(psBlob.Get());
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psoDesc.SampleMask = D3D12_DEFAULT_SAMPLE_MASK;
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
		psoDesc.NumRenderTargets = 1;
		psoDesc.RTVFormats[0] = m_format;
		psoDesc.SampleDesc.Count = 1;
		ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
	}

	// Create the command list.
	ThrowIfFailed(m_device->CreateCommandList(
		0, D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get(),
		IID_PPV_ARGS(&m_commandList)));
	ThrowIfFailed(m_commandList->Close());

	// Create the vertex buffer.
	{
		Vertex vertices[] = {
			{{-0.4f,-0.7f, 0.0f}, {0.0f, 1.0f}} , //¶‰º
			{{-0.4f, 0.7f, 0.0f}, {0.0f, 0.0f}} , //¶ã
			{{ 0.4f,-0.7f, 0.0f}, {1.0f, 1.0f}} , //‰E‰º
			{{ 0.4f, 0.7f, 0.0f}, {1.0f, 0.0f}} , //‰Eã
		};
		const UINT vertexBufferSize = sizeof(vertices);
		auto vertexHeapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		auto vertexResDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&vertexHeapProp,
			D3D12_HEAP_FLAG_NONE,
			&vertexResDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(m_vertexBuffer.ReleaseAndGetAddressOf())));
		Vertex* vertexMap = nullptr;
		ThrowIfFailed(m_vertexBuffer->Map(0, nullptr, (void**)&vertexMap));
		std::copy(std::begin(vertices), std::end(vertices), vertexMap);
		m_vertexBuffer->Unmap(0, nullptr);
		m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
		m_vertexBufferView.SizeInBytes = vertexBufferSize;
		m_vertexBufferView.StrideInBytes = sizeof(Vertex);
	}

	// Create the index buffer.
	{
		unsigned short indices[] = {
			0, 1, 2,
			2, 1, 3
		};
		const UINT indexBufferSize = sizeof(indices);
		auto indexHeapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		auto indexResDesc = CD3DX12_RESOURCE_DESC::Buffer(indexBufferSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&indexHeapProp,
			D3D12_HEAP_FLAG_NONE,
			&indexResDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_indexBuffer)));
		unsigned short* indexMap = nullptr;
		m_indexBuffer->Map(0, nullptr, (void**)&indexMap);
		std::copy(std::begin(indices), std::end(indices), indexMap);
		m_indexBuffer->Unmap(0, nullptr);
		m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
		m_indexBufferView.SizeInBytes = indexBufferSize;
		m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
	}

	// Create the SRV
	{
		D3D12_RESOURCE_DESC textureDesc = {};
		textureDesc.MipLevels = 1;
		textureDesc.Format = m_format;
		textureDesc.Width = static_cast<UINT>(m_width);
		textureDesc.Height = static_cast<UINT>(m_height);
		textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
		textureDesc.DepthOrArraySize = 1;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.SampleDesc.Quality = 0;
		textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		auto textureHeapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&textureHeapProp,
			D3D12_HEAP_FLAG_SHARED,
			&textureDesc,
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			nullptr,
			IID_PPV_ARGS(&m_textureBuffer)));

		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = textureDesc.Format;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = textureDesc.MipLevels;
		m_device->CreateShaderResourceView(m_textureBuffer.Get(), &srvDesc, m_srvHeap->GetCPUDescriptorHandleForHeapStart());

		// Share Texture Buffer resources with CUDA
		{
			HANDLE sharedHandle{};

			SECURITY_ATTRIBUTES secAttr{};
			ThrowIfFailed(m_device->CreateSharedHandle(m_textureBuffer.Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));

			const auto texAllocInfo = m_device->GetResourceAllocationInfo(m_nodeMask, 1, &textureDesc);

			cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
			cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
			cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
			cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
			cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
			cudaImportExternalMemory(&m_externalMemory_tex, &cuExtmemHandleDesc);
			CloseHandle(sharedHandle);

			cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
			cuExtmemMipDesc.extent = make_cudaExtent(textureDesc.Width, textureDesc.Height, 0);
			cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
			cuExtmemMipDesc.numLevels = 1;
			cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;

			cudaExternalMemoryGetMappedMipmappedArray(&m_cuMipArray_tex, m_externalMemory_tex, &cuExtmemMipDesc);
			cudaGetMipmappedArrayLevel(&m_cuArray_tex, m_cuMipArray_tex, 0);
		}
	}

	// Create synchronization objects and wait until assets have been uploaded to the GPU.
	{
		ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex],
			D3D12_FENCE_FLAG_SHARED,
			IID_PPV_ARGS(&m_fence)));

		cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;

		memset(&externalSemaphoreHandleDesc, 0,
			sizeof(externalSemaphoreHandleDesc));
		SECURITY_ATTRIBUTES secAttr{};
		LPCWSTR name = NULL;
		HANDLE sharedHandle;
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
		m_device->CreateSharedHandle(m_fence.Get(), &secAttr, GENERIC_ALL, name, &sharedHandle);
		externalSemaphoreHandleDesc.handle.win32.handle = (void*)sharedHandle;
		externalSemaphoreHandleDesc.flags = 0;

		cudaImportExternalSemaphore(&m_externalSemaphore, &externalSemaphoreHandleDesc);
		m_fenceValues[m_frameIndex]++;

		// Create an event handle to use for frame synchronization.
		m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (m_fenceEvent == nullptr) {
			ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
		}

		WaitForGpu();
	}
}

void DxApplication::OnRender()
{
	// Record all the commands we need to render the scene into the command list.
	PopulateCommandList();

	// Execute the command list.
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	// Present the frame.
	//ThrowIfFailed(m_swapChain->Present(1, 0));
	m_frameIndex = ++m_frameIndex % FrameCount;

	// Schedule a Signal command in the queue.
	const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
	ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

	MoveToNextFrame();
}

void DxApplication::OnDestroy()
{
	WaitForGpu();
	cudaDestroyExternalSemaphore(m_externalSemaphore);
	for (int i = 0; i < FrameCount; i++)cudaDestroyExternalMemory(m_externalMemory_out[i]);
	cudaDestroyExternalMemory(m_externalMemory_tex);
	CloseHandle(m_fenceEvent);
}

void DxApplication::SetPtr(void* in, void* out, cudaStream_t stream)
{
	m_tex_ptr = in;
	m_out_ptr = out;
	m_streamInOut = stream;
}

void DxApplication::PopulateCommandList()
{
	ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
	ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

	m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

	// Set necessary state.
	m_commandList->RSSetViewports(1, &m_viewport);
	m_commandList->RSSetScissorRects(1, &m_scissorRect);

	//// Indicate that the back buffer will be used as a render target.
	//m_commandList->ResourceBarrier(
	//	1, &CD3DX12_RESOURCE_BARRIER::Transition(
	//		m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT,
	//		D3D12_RESOURCE_STATE_RENDER_TARGET));

	ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
	m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	m_commandList->SetGraphicsRootDescriptorTable(0, m_srvHeap->GetGPUDescriptorHandleForHeapStart());

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
		m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex,
		m_rtvDescriptorSize);
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	// Record commands.
	m_commandList->ClearRenderTargetView(rtvHandle, m_clearColor, 0, nullptr);
	m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
	m_commandList->IASetIndexBuffer(&m_indexBufferView);
	m_commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);

	//// Indicate that the back buffer will now be used to present.
	//m_commandList->ResourceBarrier(
	//	1, &CD3DX12_RESOURCE_BARRIER::Transition(
	//		m_renderTargets[m_frameIndex].Get(),
	//		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	ThrowIfFailed(m_commandList->Close());
}

void DxApplication::MoveToNextFrame()
{
	const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
	cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
	memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

	externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
	externalSemaphoreWaitParams.flags = 0;

	cudaEventRecord(m_event1, m_streamInOut);

	cudaWaitExternalSemaphoresAsync(
		&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun);

	cudaStreamWaitEvent(m_streamToRun, m_event1);
	UpdateCudaTextureData();
	UpdateCudaOutData();

	cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
	memset(&externalSemaphoreSignalParams, 0, sizeof(externalSemaphoreSignalParams));
	m_fenceValues[m_frameIndex] = currentFenceValue + 1;
	externalSemaphoreSignalParams.params.fence.value = m_fenceValues[m_frameIndex];
	externalSemaphoreSignalParams.flags = 0;

	cudaSignalExternalSemaphoresAsync(
		&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun);

	cudaEventRecord(m_event2, m_streamToRun);
	cudaStreamWaitEvent(m_streamInOut, m_event2);

	// Update the frame index.
	//m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
		ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex],
			m_fenceEvent));
		WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
	}

	// Set the fence value for the next frame.
	m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}

void DxApplication::WaitForGpu()
{
	// Schedule a Signal command in the queue.
	ThrowIfFailed(
		m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

	// Wait until the fence has been processed.
	ThrowIfFailed(
		m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
	WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

	// Increment the fence value for the current frame.
	m_fenceValues[m_frameIndex]++;
}

void DxApplication::UpdateCudaTextureData()
{
	if (!m_tex_ptr) return;
	cudaMemcpy2DToArrayAsync(
		m_cuArray_tex, 0, 0, m_tex_ptr,
		m_width * sizeof(float4), m_width * sizeof(float4), m_height, cudaMemcpyDeviceToDevice,
		m_streamToRun);
}

void DxApplication::UpdateCudaOutData()
{
	if (!m_tex_ptr) return;
	cudaMemcpy2DFromArrayAsync(
		m_out_ptr, m_width * sizeof(float4), m_cuArray_out[m_frameIndex],
		0, 0, m_width * sizeof(float4), m_height, cudaMemcpyDeviceToDevice,
		m_streamToRun);
}
