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
            new ImageTool { Name = "Background Removal", Description = "Automatically removes the background from an image, leaving only the main subject. Useful for product photos or profile pictures.", ScriptPath = "PythonScripts/background_removal.py" },
            new ImageTool { Name = "Color Space Transformation", Description = "Converts an image from one color space (e.g., RGB) to another (e.g., grayscale or HSV), allowing for advanced color-based processing.", ScriptPath = "PythonScripts/color_space_transformation.py" },
            new ImageTool { Name = "Contour Detection", Description = "Identifies the outlines of objects within an image, helpful for shape analysis, object segmentation, or feature extraction.", ScriptPath = "PythonScripts/contour_detection.py" },
            new ImageTool { Name = "Face Detection", Description = "Detects human faces within an image using machine learning models. Commonly used in surveillance, photo tagging, and biometrics.", ScriptPath = "PythonScripts/face_detection.py" },
            new ImageTool { Name = "Resize", Description = "Changes the width and height of an image to a new resolution. Helps with standardizing image sizes or preparing inputs for AI models.", ScriptPath = "PythonScripts/resize.py" },
            new ImageTool { Name = "Text Recognition", Description = "Extracts text from an image using Optical Character Recognition (OCR). Useful for digitizing documents, signs, or license plates.", ScriptPath = "PythonScripts/text_recognition.py" }
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