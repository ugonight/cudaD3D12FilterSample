#pragma once

#include <cuda_runtime.h>
#include <wrl.h>
#include "d3dx12.h"
#include <DirectXMath.h>

using Microsoft::WRL::ComPtr;

struct Vertex
{
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT2 uv;
};

class DxApplication
{
public:
	DxApplication(UINT width, UINT height);

	void OnInit();
	void OnRender();
	void OnDestroy();
	void SetPtr(void* in, void* out, cudaStream_t stream);

private:
	static const UINT FrameCount = 2;

	// Pipeline objects.
	D3D12_VIEWPORT m_viewport;
	CD3DX12_RECT m_scissorRect;
	//ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12Device> m_device;
	ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
	ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
	ComPtr<ID3D12CommandQueue> m_commandQueue;
	ComPtr<ID3D12RootSignature> m_rootSignature;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12DescriptorHeap> m_srvHeap;
	ComPtr<ID3D12PipelineState> m_pipelineState;
	ComPtr<ID3D12GraphicsCommandList> m_commandList;
	UINT m_rtvDescriptorSize;

	// App resources.
	ComPtr<ID3D12Resource> m_vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
	ComPtr<ID3D12Resource> m_indexBuffer;
	D3D12_INDEX_BUFFER_VIEW m_indexBufferView;
	ComPtr<ID3D12Resource> m_textureBuffer;

	// Synchronization objects.
	UINT m_frameIndex;
	HANDLE m_fenceEvent;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fenceValues[FrameCount];

	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	cudaExternalSemaphore_t m_externalSemaphore;
	cudaExternalMemory_t m_externalMemory_out[FrameCount];
	cudaMipmappedArray_t m_cuMipArray_out[FrameCount]{};
	cudaArray_t m_cuArray_out[FrameCount]{};
	cudaExternalMemory_t m_externalMemory_tex;
	cudaMipmappedArray_t m_cuMipArray_tex{};
	cudaArray_t m_cuArray_tex{};
	cudaStream_t m_streamToRun;
	LUID m_dx12deviceluid;
	UINT m_cudaDeviceID;
	UINT m_nodeMask;
	void* m_out_ptr, * m_tex_ptr;
	cudaStream_t m_streamInOut;
	cudaEvent_t m_event1, m_event2;

	UINT m_width;
	UINT m_height;

	DXGI_FORMAT m_format;
	float m_clearColor[4];

	void LoadPipeline();
	void InitCuda();
	void LoadAssets();
	void PopulateCommandList();
	void MoveToNextFrame();
	void WaitForGpu();

	void UpdateCudaTextureData();
	void UpdateCudaOutData();
};

