; The Vibe Hash setup wizard for Windows (Inno Setup 6): VibeHash-Setup.exe, what the website's download button gives.
; It asks the language, whether to install for everyone or just this user, where, and about a desktop icon and
; starting at login; puts the launcher (launcher/vibehash_launcher.py built into VibeHash.exe) there with Start menu
; and desktop shortcuts; registers Vibe Hash in Settings > Apps with a real uninstaller; and starts it. The launcher
; then does what it always does on first run (uv, a Python of its own, Vibe Hash from PyPI, in %LOCALAPPDATA%\VibeHash
; for the user running it), but makes no shortcuts or Apps entry of its own: it sees the uninstaller beside it.
;
; Uninstall (Settings > Apps, or the Start menu's entry): stops a running Vibe Hash and removes the runtime the
; launcher made (`VibeHash.exe --uninstall --yes`), removes the program, shortcuts and start-at-login, then asks
; whether to delete the wallet and settings too (~/.vibehash; kept unless said so).
;
; Built by .github/workflows/launcher.yml on a version tag (ISCC /DMyAppVersion=1.2.3 installer\VibeHash.iss, with
; dist\VibeHash.exe already built by PyInstaller); by hand: the same from the repo's root. Not code-signed yet.

#define MyAppName "Vibe Hash"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppExe "VibeHash.exe"
#define MyAppUrl "https://vibehash.org"

[Setup]
AppId={{7D1B6A2E-5C3F-4A8E-9B1D-2F6E0C4A7B93}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Vibe Hash
AppPublisherURL={#MyAppUrl}
AppSupportURL={#MyAppUrl}
AppUpdatesURL={#MyAppUrl}
; Program Files for everyone, or this user's own programs folder: the wizard asks (PrivilegesRequiredOverridesAllowed)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ShowLanguageDialog=yes
WizardStyle=modern
SetupIconFile=..\launcher\VibeHash.ico
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=VibeHash-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "catalan"; MessagesFile: "compiler:Languages\Catalan.isl"
Name: "czech"; MessagesFile: "compiler:Languages\Czech.isl"
Name: "danish"; MessagesFile: "compiler:Languages\Danish.isl"
Name: "dutch"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "finnish"; MessagesFile: "compiler:Languages\Finnish.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "hungarian"; MessagesFile: "compiler:Languages\Hungarian.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "norwegian"; MessagesFile: "compiler:Languages\Norwegian.isl"
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "portuguese"; MessagesFile: "compiler:Languages\Portuguese.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"

[CustomMessages]
english.StartAtLogin=Start Vibe Hash when I log in (mining in the background)
english.DeleteWallet=Also delete your wallet, its recovery words and your settings in%n%1?%n%nWithout the recovery words the coins in that wallet are gone for good. Choose No to keep them.
english.RuntimeNote=Vibe Hash keeps its own Python and the AI libraries in %1 for each user who runs it; the uninstaller removes yours.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startup"; Description: "{cm:StartAtLogin}"; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExe}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Comment: "Mining that trains real AI models"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Registry]
; Start at login: a Run value for this user (the launcher is a windowed program: nothing flashes by). Gone with the uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VibeHash"; ValueData: """{app}\{#MyAppExe}"" -- --no-window"; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop a running Vibe Hash, take out its start-at-login entry and remove the runtime it made for this user.
Filename: "{app}\{#MyAppExe}"; Parameters: "--uninstall --yes"; Flags: runhidden waituntilterminated; RunOnceId: "VibeHashRuntime"

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\VibeHash"

[Code]
// After everything else is gone: the wallet and settings stay unless the user says otherwise (never in a silent run).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Home: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Home := GetEnv('VIBEHASH_HOME');
    if Home = '' then
      Home := ExpandConstant('{%USERPROFILE}') + '\.vibehash';
    if DirExists(Home) and not UninstallSilent then
      if MsgBox(FmtMessage(CustomMessage('DeleteWallet'), [Home]), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(Home, True, True, True);
  end;
end;
