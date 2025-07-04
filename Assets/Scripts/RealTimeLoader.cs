using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Intel.RealSense;
using System;
using System.Runtime.InteropServices;

public class RealTimeLoader : MonoBehaviour
{
    // Inspector References
    public RsStreamTextureRenderer colorRenderer;
    public RsStreamTextureRenderer depthRenderer;
    public RealSenseSender networkSender;
    public RawImage previewImage;
    public TextMeshProUGUI resultText;

    // Runtime Data
    private Texture2D colorTexture;
    private ushort[] depthData;
    private byte[] depthBuffer;
    private FrameQueue colorQueue;
    private FrameQueue depthQueue;

    void Start()
    {
        // Initialize frame queues
        colorQueue = new FrameQueue(1);
        depthQueue = new FrameQueue(1);

        // Subscribe to frame providers
        if (colorRenderer != null)
        {
            colorRenderer.textureBinding.AddListener(OnNewColorTexture);
        }

        if (depthRenderer != null)
        {
            depthRenderer.Source.OnNewSample += OnNewDepthFrame;
        }
    }

    void OnDestroy()
    {
        if (colorTexture != null)
        {
            Destroy(colorTexture);
            colorTexture = null;
        }

        if (colorQueue != null)
        {
            colorQueue.Dispose();
            colorQueue = null;
        }

        if (depthQueue != null)
        {
            depthQueue.Dispose();
            depthQueue = null;
        }
    }


    private void OnNewColorTexture(Texture texture)
    {
        colorTexture = (Texture2D)texture;
    }

    private void OnNewDepthFrame(Frame frame)
    {
        if (frame.IsComposite)
        {
            using (var frames = frame.As<FrameSet>())
            using (var depthFrame = frames.FirstOrDefault<DepthFrame>(Stream.Depth, Format.Z16))
            {
                if (depthFrame != null)
                {
                    depthQueue.Enqueue(depthFrame.Clone());
                }
            }
            return;
        }

        // Fixed the Is() method usage - check stream and format separately
        using (var profile = frame.Profile)
        {
            if (profile.Stream == Stream.Depth && profile.Format == Format.Z16)
            {
                depthQueue.Enqueue(frame.Clone());
            }
        }
    }

    void Update()
    {
        ProcessDepthFrame();
    }

    private void ProcessDepthFrame()
    {
        DepthFrame depthFrame;
        if (depthQueue != null && depthQueue.PollForFrame<DepthFrame>(out depthFrame))
        {
            using (depthFrame)
            {
                int pixelCount = depthFrame.Width * depthFrame.Height;
                int byteCount = pixelCount * sizeof(ushort);

                if (depthData == null || depthData.Length != pixelCount)
                    depthData = new ushort[pixelCount];

                if (depthBuffer == null || depthBuffer.Length < byteCount)
                    depthBuffer = new byte[byteCount];

                Marshal.Copy(depthFrame.Data, depthBuffer, 0, byteCount);
                Buffer.BlockCopy(depthBuffer, 0, depthData, 0, byteCount);
            }
        }
    }

    public void CaptureAndSend()
    {
        if (colorTexture == null || depthData == null)
        {
            resultText.text = "Error: No frame data available";
            return;
        }

        try
        {
            byte[] rgbBytes = colorTexture.EncodeToPNG();
            byte[] depthBytes = new byte[depthData.Length * sizeof(ushort)];
            Buffer.BlockCopy(depthData, 0, depthBytes, 0, depthBytes.Length);

            networkSender.SendBytes(rgbBytes, depthBytes, (result) => {
                resultText.text = result;
            });
        }
        catch (Exception ex)
        {
            resultText.text = $"Capture Error: {ex.Message}";
        }
    }
}