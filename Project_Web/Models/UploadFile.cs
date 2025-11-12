namespace PAW_Project.Models;

public class UploadFile
{
    public int Id { get; set; }
    public Guid Token { get; set; } = Guid.NewGuid();
    public string FileName { get; set; } = string.Empty; // This is the filename used on the application's server
    
    public string OriginalFileName { get; set; } = string.Empty; // This is the filename shown to the user.
    public string? UserId { get; set; } = null;
    
    public bool IsTemp { get; set; } = true;
    public DateTime AddedDate { get; set; } = DateTime.UtcNow;
    
    public ApplicationUser? User { get; set; }
}