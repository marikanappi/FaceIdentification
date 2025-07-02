using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuManager : MonoBehaviour
{
    public void LoadScene1()
    {
        SceneManager.LoadScene("Static");
    }

    public void LoadScene2()
    {
        SceneManager.LoadScene("RealTime");
    }
}
