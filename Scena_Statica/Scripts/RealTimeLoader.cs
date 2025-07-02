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
    private byte[] colorBuffer;
    private byte[] depthBuffer; // For intermediate depth copy
    private int lastColorWidth = -1;
    private int lastColorHeight = -1;

    void Update()
    {
        UpdateFrames();
    }

    void UpdateFrames()
    {
        // Color Frame processing
        using (var clone = colorRenderer?.LatestFrame?.Clone() as VideoFrame)  //changed colorFrame with clone to avoid disposal
        {
            try
            {
                if (clone != null)
                {
                    if (colorTexture == null ||
                        clone.Width != lastColorWidth ||
                        clone.Height != lastColorHeight)
                    {
                        colorTexture = new Texture2D(
                            clone.Width,
                            clone.Height,
                            TextureFormat.BGRA32,
                            false);
                        lastColorWidth = clone.Width;
                        lastColorHeight = clone.Height;

                        colorBuffer = new byte[clone.Stride * clone.Height];
                    }

                    Marshal.Copy(clone.Data, colorBuffer, 0, colorBuffer.Length);
                    colorTexture.LoadRawTextureData(colorBuffer);
                    colorTexture.Apply();
                    previewImage.texture = colorTexture;
                }
            }
            catch (ObjectDisposedException)
            {
                Debug.LogWarning("Color frame disposed during processing - skipped");
            }
        }

        // Depth Frame processing
        if (depthRenderer?.LatestFrame is DepthFrame depthFrame)
        {
            try
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
            catch (ObjectDisposedException)
            {
                Debug.LogWarning("Depth frame disposed during processing - skipped");
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

    void OnDestroy()
    {
        if (colorTexture != null)
        {
            Destroy(colorTexture);
            colorTexture = null;
        }
    }
}
