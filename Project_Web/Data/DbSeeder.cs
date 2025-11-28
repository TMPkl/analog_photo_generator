using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using PAW_Project.Models;

namespace PAW_Project.Data;

public static class DbSeeder
{
    public static async Task SeedImageToolsAsync(IServiceProvider serviceProvider)
    {
        using var scope = serviceProvider.CreateScope();
        var context = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var toolsToSeed = new List<ImageTool>
        {
            new ImageTool { Name = "Grain Generator", Description = "Add multiscale grain to an image (expects BGR input). The script converts to HLS, adds grain and saves output.", ScriptPath = "PythonScripts/grain.py" },
        };

        foreach (var tool in toolsToSeed)
        {
            var exists = await context.ImageTools.AnyAsync(t => t.Name == tool.Name);
            if (!exists)
            {
                context.ImageTools.Add(tool);
            }
        }

        await context.SaveChangesAsync();
    }


}