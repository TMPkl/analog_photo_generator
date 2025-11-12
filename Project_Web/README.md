# Image helper

This project is a web-based platform for uploading and processing images using a variety of server-side tools.
Users can upload image files, select from different processing options (such as background removal, object detection, resizing, etc.),
and receive the processed results directly in the browser.

Some tools also provide additional output, like detected text.
The system supports temporary and persistent file storage, and seamless integration with Python-based image processing scripts.

## How to run

This project was built in JetBrains Rider on Ubuntu.

### Requirements

- .NET 8 SDK or later
- MySQL (running locally)
- EF Core CLI tools (dotnet ef)

### JetBrains Rider

- Open the solution file
- Open _appsettings.json_ and edit the connection settings **ConnectionStrings:DefaultConnection** to match your MySQL connection
- Run the migrations running this in the terminal: `dotnet ef database update`
- In the command line, run `python3 ./PythonScripts/activate_venv.py`
  - This requires that a python 3.10 version is installed and usable on the device.
- Build and Run the project **(Shift + F10)**

### Visual Studio

- Open the solution file
- Open _appsettings.json_ and edit the connection settings **ConnectionStrings:DefaultConnection** to match your MySQL connection
- Run the migrations running this in the **Package Manager Console**: `Update-Database`
- In the command line, run `python3 ./PythonScripts/activate_venv.py`
  - This requires that a python 3.10 version is installed and usable on the device.
- Run the project **(F5)**

(This was not tested)

### CLI

Run the following:

```
dotnet ef database update
python3 ./PythonScripts/activate_venv.py
dotnet run
```

(This was not tested)

### For testing purposes:

The development mailservice used for the application is [smtp4dev](https://github.com/rnwood/smtp4dev), which provides
mock mail sending utility without actually launching mails outside of the development environment.

It can be run using this command (SMTP port 2525)

```
docker run -d -p 3000:80 -p 2525:25 rnwood/smtp4dev
```

After which the Web UI can be accessed at the following link [http://localhost:3000](http://localhost:3000)
