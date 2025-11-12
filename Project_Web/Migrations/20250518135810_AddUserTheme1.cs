using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace PAW_Project.Migrations
{
    /// <inheritdoc />
    public partial class AddUserTheme1 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "PreferedTheme",
                table: "AspNetUsers",
                newName: "PreferredTheme");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "PreferredTheme",
                table: "AspNetUsers",
                newName: "PreferedTheme");
        }
    }
}
