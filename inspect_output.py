import json, os

fp = r'c:\Users\Krishna\AppData\Roaming\Code\User\workspaceStorage\fae853281beb66b416b2cbf4bd2825e5\GitHub.copilot-chat\chat-session-resources\d506fc8b-6c94-40ed-87b3-b8bac3542c8b\toolu_bdrk_01RaeECjecjVcW2B4QkkckKr__vscode-1771908489173\content.json'
data = json.load(open(fp, encoding='utf-8'))

pth_files = [f['file_name'] for f in data['files'] if f['file_name'].endswith('.pth')]
print(f"PTH files ({len(pth_files)}):")
for p in pth_files:
    print(f"  {p}")

best_files = [f['file_name'] for f in data['files'] if 'best' in f['file_name'].lower()]
print(f"\n'best' files ({len(best_files)}):")
for p in best_files:
    print(f"  {p}")

print(f"\nTotal files: {len(data['files'])}")

log = data.get('log', '')
if log:
    print(f"\nLog length: {len(log)}")
    print("LOG last 1500 chars:")
    print(log[-1500:])
else:
    print("\nNo log available")
