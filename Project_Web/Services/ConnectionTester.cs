using PAW_Project.Data;

namespace PAW_Project.Services;

public class ConnectionTester
{
    private readonly AppDbContext _context;

    public ConnectionTester(AppDbContext context)
    {
        _context = context;
    }

    public async Task<bool> TestConnectionAsync()
    {
        try
        {
            return await _context.Database.CanConnectAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Connection failed: {ex.Message}");
            return false;
        }
    }
}
