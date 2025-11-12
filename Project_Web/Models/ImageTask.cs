using System.ComponentModel.DataAnnotations;

namespace PAW_Project.Models;

public class ImageTask
{
    public int Id { get; set; }
    public string OutputPath { get; set; } = string.Empty;
    public int FileId { get; set; }
    
    public int? ImageToolId { get; set; }
    
    public UploadFile File { get; set; } = null!;
    public ImageTool ImageTool { get; set; } = null!;
}