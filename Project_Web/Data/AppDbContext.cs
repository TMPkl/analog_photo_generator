using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using PAW_Project.Models;

namespace PAW_Project.Data;

public class AppDbContext : IdentityDbContext<ApplicationUser>
{
    public DbSet<UploadFile> UploadFiles { get; set; } = null!;
    public DbSet<ImageTask> ImageTasks { get; set; } = null!;
    public DbSet<ImageTool> ImageTools { get; set; } = null!;
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

}