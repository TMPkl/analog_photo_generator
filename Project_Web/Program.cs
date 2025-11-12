using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using PAW_Project.Data;
using PAW_Project.Models;
using PAW_Project.Services;

namespace PAW_Project;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Add services to the container.
        builder.Services.AddControllersWithViews();
        
        // Register the MySQL Database Connection
        builder.Services.AddDbContext<AppDbContext>(options =>
            options.UseMySql(builder.Configuration.GetConnectionString("DefaultConnection"),
                new MySqlServerVersion(new Version(8, 0, 41)))); 
        
        // Setup EF Core Identity and set it up to connect to MySQL database
        builder.Services.AddIdentity<ApplicationUser, IdentityRole>()
            .AddEntityFrameworkStores<AppDbContext>()
            .AddDefaultTokenProviders();
        
        builder.Services.Configure<IdentityOptions>(options =>
        {
            options.User.RequireUniqueEmail = true;
        });
        
        // Enable cookies for login and redirect Login and Unauthorized pages
        builder.Services.ConfigureApplicationCookie(options =>
        {
            options.LoginPath = "/Account/Login";
            options.AccessDeniedPath = "/Account/AccessDenied";
        });
        
        builder.Services.AddScoped<ConnectionTester>();

        builder.Services.AddSession();



        var app = builder.Build();
        

        // Configure the HTTP request pipeline.
        if (!app.Environment.IsDevelopment())
        {
            app.UseExceptionHandler("/Home/Error");
            // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
            app.UseHsts();
        }

        app.UseHttpsRedirection();
        app.UseRouting();
        
        // Enforce Authentication on some pages
        app.UseAuthentication();
        app.UseAuthorization();
        app.UseSession();

        app.MapStaticAssets();
        app.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}")
            .WithStaticAssets();
        
        // Call the DBSeeder to create Roles in the database provided they don't exist
        using (var scope = app.Services.CreateScope())
        {
            DbSeeder.SeedImageToolsAsync(scope.ServiceProvider).Wait();
        }

        app.Run();
    }
}