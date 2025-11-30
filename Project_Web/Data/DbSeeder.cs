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
            new ImageTool { Name = "Halation Generator", Description = "Detect bright light sources and add halation (halo) effect.", ScriptPath = "PythonScripts/haliation.py" },
            new ImageTool { Name = "Apply Look-Up Table", Description = "Apply 3D Look-Up Table (LUT) to an image for an analog effect.", ScriptPath = "PythonScripts/lut.py" },
            new ImageTool { Name = "Apply AB Transform", Description = "Run AB analog transform using Generative Adversarial Networks.", ScriptPath = "PythonScripts/gan_ab.py" },
            new ImageTool { Name = "Apply LAB Transform", Description = "Run LAB analog transform using Generative Adversarial Networks.", ScriptPath = "PythonScripts/gan_lab.py" },
            new ImageTool { Name = "GAN Analog Pipeline", Description = "Run a pipeline that runs a GAN model, then adds halation and grain to the image.", ScriptPath = "PythonScripts/pipeline_gan.py" },
            new ImageTool { Name = "LUT Analog Pipeline", Description = "Run a pipeline that runs a LUT transform, then adds halation and grain to the image.", ScriptPath = "PythonScripts/pipeline_lut.py" }
            
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