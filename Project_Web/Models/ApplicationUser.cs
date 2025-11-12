using Microsoft.AspNetCore.Identity;

namespace PAW_Project.Models;

public class ApplicationUser : IdentityUser
{
    public string? PreferredTheme { get; set; }
}