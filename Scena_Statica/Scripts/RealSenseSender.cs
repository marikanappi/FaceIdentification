using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class RealSenseSender : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int serverPort = 5005;
    public int sendTimeoutMs = 15000;
    public int receiveTimeoutMs = 15000;

    private TcpClient client = null;
    private NetworkStream stream = null;
    private bool isConnected = false;

    public async void Connect(Action<bool> callback)
    {
        try
        {
            if (isConnected) return;
            
            client = new TcpClient();
            var connectTask = client.ConnectAsync(serverIP, serverPort);
            
            await Task.WhenAny(connectTask, Task.Delay(5000));
            if (!client.Connected)
            {
                callback?.Invoke(false);
                return;
            }

            stream = client.GetStream();
            stream.WriteTimeout = sendTimeoutMs;
            stream.ReadTimeout = receiveTimeoutMs;
            isConnected = true;
            callback?.Invoke(true);
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection Error: {e.Message}");
            callback?.Invoke(false);
        }
    }

    public void ConnectFromUI() {
    Connect((success) => {
        Debug.Log(success ? "Connected!" : "Failed");
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
            // 2. Invia RGB
            await SendChunk(stream, BitConverter.GetBytes(rgbBytes.Length), "RGB Size");
            await SendChunk(stream, rgbBytes, "RGB Data");

            // 3. Invia Depth
            await SendChunk(stream, BitConverter.GetBytes(depthBytes.Length), "Depth Size");
            await SendChunk(stream, depthBytes, "Depth Data");

            // 4. Ricevi risposta
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
            stream?.Close();     // Closes the network stream first
            stream?.Dispose();   // Ensures resources are freed
            client?.Close();      // Then close the TCP client
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