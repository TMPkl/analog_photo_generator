using System.Diagnostics;
using System.Security.Claims;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PAW_Project.Data;
using PAW_Project.Models;
using PAW_Project.ViewModels;

namespace PAW_Project.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly AppDbContext _context;
    

    public HomeController(ILogger<HomeController> logger, AppDbContext context)
    {
        _logger = logger;
        _context = context;

    }

    public async Task<IActionResult> Index(Guid? file)
    {
        var model = new HomeViewModel()
        {
            ImageTools = await _context.ImageTools.ToListAsync(),
        };
        await CleanupOldTempFilesAsync();
        
        // if a file is set in the GET request, load it for the user
        if (file.HasValue)
        {
            var upload = await _context.UploadFiles.Where(f => f.Token == file).FirstOrDefaultAsync();

            if (upload != null)
            {
                model.UploadedFilePath = Url.Content($"~/uploads/{upload.FileName}");
                model.FileToken = upload.Token;
            }
        }
        
        return View(model);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
    
    private async Task CleanupOldTempFilesAsync()
    {
        var random = new Random();
        if (random.NextDouble() > 0.2)
            return; 

        var sixHoursAgo = DateTime.UtcNow.AddHours(-6);
        var oldTempFiles = _context.UploadFiles
            .Where(f => f.AddedDate < sixHoursAgo)
            .ToList();

        if (oldTempFiles.Count == 0) return;

        var uploadsPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");

        foreach (var file in oldTempFiles)
        {
            Console.WriteLine(file.FileName);
            var filePath = Path.Combine(uploadsPath, file.FileName);
            if (System.IO.File.Exists(filePath))
            {
                try
                {
                    System.IO.File.Delete(filePath);
                }
                catch (Exception ex)
                {
                }
            }

            _context.UploadFiles.Remove(file);
        }

        await _context.SaveChangesAsync();
    }

}