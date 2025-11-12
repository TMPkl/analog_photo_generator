using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace PAW_Project.Migrations
{
    /// <inheritdoc />
    public partial class AddUploadFileFields111 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_UploadFiles_AspNetUsers_UserId",
                table: "UploadFiles");

            migrationBuilder.AlterColumn<string>(
                name: "UserId",
                table: "UploadFiles",
                type: "varchar(255)",
                nullable: true,
                oldClrType: typeof(string),
                oldType: "varchar(255)")
                .Annotation("MySql:CharSet", "utf8mb4")
                .OldAnnotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.AddColumn<bool>(
                name: "IsTemp",
                table: "UploadFiles",
                type: "tinyint(1)",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<string>(
                name: "OriginalFileName",
                table: "UploadFiles",
                type: "longtext",
                nullable: false)
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.AddColumn<Guid>(
                name: "Token",
                table: "UploadFiles",
                type: "char(36)",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"),
                collation: "ascii_general_ci");

            migrationBuilder.AddForeignKey(
                name: "FK_UploadFiles_AspNetUsers_UserId",
                table: "UploadFiles",
                column: "UserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_UploadFiles_AspNetUsers_UserId",
                table: "UploadFiles");

            migrationBuilder.DropColumn(
                name: "IsTemp",
                table: "UploadFiles");

            migrationBuilder.DropColumn(
                name: "OriginalFileName",
                table: "UploadFiles");

            migrationBuilder.DropColumn(
                name: "Token",
                table: "UploadFiles");

            migrationBuilder.UpdateData(
                table: "UploadFiles",
                keyColumn: "UserId",
                keyValue: null,
                column: "UserId",
                value: "");

            migrationBuilder.AlterColumn<string>(
                name: "UserId",
                table: "UploadFiles",
                type: "varchar(255)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "varchar(255)",
                oldNullable: true)
                .Annotation("MySql:CharSet", "utf8mb4")
                .OldAnnotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.AddForeignKey(
                name: "FK_UploadFiles_AspNetUsers_UserId",
                table: "UploadFiles",
                column: "UserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
