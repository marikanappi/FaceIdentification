using UnityEngine;
using UnityEngine.UI;
using System.IO;
using TMPro;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class ImageLoader : MonoBehaviour
{
    public RealSenseSender sender;
    public TextMeshProUGUI resultText;
    public RawImage rgbPreviewImage;    // Assign in Inspector
    public RawImage depthPreviewImage;  // Assign in Inspector
    public TextMeshProUGUI rgbInfoText; // Optional: Show RGB info
    public TextMeshProUGUI depthInfoText; // Optional: Show depth info

    private byte[] rgbBytes;
    private byte[] depthBytes;
    private Texture2D rgbTexture;
    private Texture2D depthTexture;

    public void LoadRGB()
    {
#if UNITY_EDITOR
        string path = EditorUtility.OpenFilePanel("Seleziona immagine RGB (PNG)", "", "png");
        if (!string.IsNullOrEmpty(path))
        {
            rgbBytes = File.ReadAllBytes(path);
            Debug.Log("Immagine RGB caricata: " + path);

            // Create and display texture
            rgbTexture = new Texture2D(2, 2);
            if (rgbTexture.LoadImage(rgbBytes))
            {
                rgbPreviewImage.texture = rgbTexture;

                // Update info text
                if (rgbInfoText != null)
                    rgbInfoText.text = $"RGB: {rgbTexture.width}x{rgbTexture.height}";
            }
        }
#endif
    }

    public void LoadDepth()
    {
#if UNITY_EDITOR
        string path = EditorUtility.OpenFilePanel("Seleziona file Depth RAW", "", "raw");
        if (!string.IsNullOrEmpty(path))
        {
            depthBytes = File.ReadAllBytes(path);
            Debug.Log("File RAW caricato: " + path);

            // Assuming depth is 16-bit grayscale, 640x480 (adjust as needed)
            int width = 640;
            int height = 480;
            depthTexture = new Texture2D(width, height, TextureFormat.R16, false);
            depthTexture.LoadRawTextureData(depthBytes);
            depthTexture.Apply();

            depthPreviewImage.texture = depthTexture;

            // Update info text
            if (depthInfoText != null)
                depthInfoText.text = $"Depth: {width}x{height}";
        }
#endif
    }

    public void SendToPython()
    {
        resultText.text = "Waiting for identification...";
        if (rgbBytes == null || depthBytes == null)
        {
            Debug.LogWarning("Carica sia RGB che RAW prima di inviare.");
            return;
        }

        sender.SendBytes(rgbBytes, depthBytes, (identity) =>
        {
            Debug.Log("Risultato ricevuto: " + identity);
            resultText.text = "Identità: " + identity;
        });
    }

    void OnDestroy()
    {
        // Clean up textures
        if (rgbTexture != null)
            Destroy(rgbTexture);
        if (depthTexture != null)
            Destroy(depthTexture);
    }
}