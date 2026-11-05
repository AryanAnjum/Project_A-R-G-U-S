; A.R.G.U.S. Windows installer (Inno Setup 6). Built by installer\build_installer.ps1, which passes:
;   /DStaging=<installer\build\ARGUS>  /DRedist=<installer\build\redist>  /DOutDir=<dist>  /DAppVersion=1.0.0

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6F3A2C1E-8B4D-4E7A-9C2F-A5D1E0B3C742}
AppName=A.R.G.U.S.
AppVersion={#AppVersion}
AppVerName=A.R.G.U.S. {#AppVersion}
AppPublisher=Aryan Anjum
AppPublisherURL=https://github.com/AryanAnjum/Project_A-R-G-U-S
; Per-user install: no admin rights needed, and the app can write its logs next to itself.
DefaultDirName={localappdata}\Programs\ARGUS
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
OutputDir={#OutDir}
OutputBaseFilename=ARGUS-Setup
SetupIconFile={#Staging}\argus.ico
UninstallDisplayIcon={app}\argus.ico
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; About 12 GB of files: split into Setup.exe + 2 GB .bin slices. Model weights barely compress,
; so fast compression keeps the build time reasonable at almost the same size.
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=max

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Staging}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Redist}\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VCRedistNeeded
Source: "{#Redist}\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: WebView2Needed

[Icons]
Name: "{autoprograms}\A.R.G.U.S."; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\src\argus_desktop.py"""; WorkingDir: "{app}\app\src"; IconFilename: "{app}\argus.ico"; Comment: "A.R.G.U.S. egocentric assistant"
Name: "{autodesktop}\A.R.G.U.S."; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\src\argus_desktop.py"""; WorkingDir: "{app}\app\src"; IconFilename: "{app}\argus.ico"; Tasks: desktopicon
Name: "{autoprograms}\A.R.G.U.S. (with console)"; Filename: "{app}\python\python.exe"; Parameters: """{app}\app\src\argus_desktop.py"""; WorkingDir: "{app}\app\src"; IconFilename: "{app}\argus.ico"; Comment: "Start A.R.G.U.S. and show its log window"

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing the Microsoft Visual C++ runtime..."; Check: VCRedistNeeded; Flags: waituntilterminated
Filename: "{tmp}\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft Edge WebView2..."; Check: WebView2Needed; Flags: waituntilterminated
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\src\argus_desktop.py"""; WorkingDir: "{app}\app\src"; Description: "Start A.R.G.U.S. now"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Python bytecode, logs and anything else the app wrote inside its own folder.
Type: filesandordirs; Name: "{app}"

[Code]
function VCRedistNeeded: Boolean;
var
  Installed: Cardinal;
begin
  Result := not (RegQueryDWordValue(HKLM32, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1));
end;

function WebView2Found(RootKey: Integer; const SubKey: String): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(RootKey, SubKey, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0');
end;

function WebView2Needed: Boolean;
begin
  Result := not (WebView2Found(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}')
    or WebView2Found(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'));
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  // nvidia-smi ships with every NVIDIA driver; without it there is almost certainly no usable GPU.
  if not FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
    Result := MsgBox('No NVIDIA graphics driver was found on this PC.' + #13#10#13#10 +
      'A.R.G.U.S. needs an NVIDIA GPU with 8 GB or more of video memory (RTX 3060 or newer recommended) ' +
      'and a recent driver (version 570 or newer).' + #13#10#13#10 +
      'Install anyway?', mbConfirmation, MB_YESNO) = IDYES;
end;
