using PAW_Project.Models;

namespace PAW_Project.ViewModels;

public class HomeViewModel
{
    public List<ImageTool> ImageTools { get; set; } = new List<ImageTool>();
    
    public string? UploadedFilePath { get; set; }
    public Guid? FileToken { get; set; }
}