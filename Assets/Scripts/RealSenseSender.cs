using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class RealSenseSender : MonoBehaviour
{
    public static RealSenseSender Instance;

    public string serverIP = "127.0.0.1";
    public int serverPort = 5005;
    public int sendTimeoutMs = 15000;
    public int receiveTimeoutMs = 15000;

    private TcpClient client = null;
    private NetworkStream stream = null;
    private bool isConnected = false;

    void Awake()
    {
        // Singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject); // Doppione? Lo elimino.
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject); // Sopravvive tra le scene
    }

    public async void Connect(Action<bool> callback)
    {
        try
        {
            if (isConnected)
            {
                Debug.Log("Già connesso.");
                callback?.Invoke(true);
                return;
            }

            client = new TcpClient();
            var connectTask = client.ConnectAsync(serverIP, serverPort);

            await Task.WhenAny(connectTask, Task.Delay(5000));
            if (!client.Connected)
            {
                Debug.LogWarning("Connessione fallita.");
                callback?.Invoke(false);
                return;
            }

            stream = client.GetStream();
            stream.WriteTimeout = sendTimeoutMs;
            stream.ReadTimeout = receiveTimeoutMs;
            isConnected = true;

            Debug.Log("Connessione stabilita.");
            callback?.Invoke(true);
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection Error: {e.Message}");
            callback?.Invoke(false);
        }
    }

    public void ConnectFromUI()
    {
        Connect((success) => {
            Debug.Log(success ? "✅ Connected!" : "❌ Failed to connect.");
        });
    }

    public async void SendBytes(byte[] rgbBytes, byte[] depthBytes, Action<string> callback)
    {
        if (!isConnected)
        {
            callback?.Invoke("Error: Not connected to server");
            return;
        }

        try
        {
            await SendChunk(stream, BitConverter.GetBytes(rgbBytes.Length), "RGB Size");
            await SendChunk(stream, rgbBytes, "RGB Data");

            await SendChunk(stream, BitConverter.GetBytes(depthBytes.Length), "Depth Size");
            await SendChunk(stream, depthBytes, "Depth Data");

            string response = await ReceiveResponse(stream);
            callback?.Invoke(response);
        }
        catch (Exception e)
        {
            Debug.LogError($"Send Error: {e.Message}");
            callback?.Invoke($"Error: {e.Message}");
            Disconnect();
        }
    }

    private async Task SendChunk(NetworkStream stream, byte[] data, string label)
    {
        Debug.Log($"Sending {label} ({data.Length} bytes)");
        await stream.WriteAsync(data, 0, data.Length);
    }

    private async Task<string> ReceiveResponse(NetworkStream stream)
    {
        byte[] buffer = new byte[1024];
        var ms = new System.IO.MemoryStream();

        while (true)
        {
            int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
            if (bytesRead == 0) break;

            ms.Write(buffer, 0, bytesRead);

            // Cerca terminatore null
            byte[] received = ms.ToArray();
            int nullPos = Array.IndexOf(received, (byte)0);
            if (nullPos >= 0)
            {
                return System.Text.Encoding.UTF8.GetString(received, 0, nullPos);
            }
        }
        return "Error: No null terminator received";
    }

    public void Disconnect()
    {
        try
        {
            stream?.Close();
            stream?.Dispose();
            client?.Close();
            client?.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogError($"Disconnect Error: {e.Message}");
        }
        finally
        {
            stream = null;
            client = null;
            isConnected = false;
        }
    }

    void OnDestroy()
    {
        Disconnect();
    }
}
