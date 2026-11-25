"""Application deployment environments. No ROS imports and no shell execution."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from robotic_inference.errors import ConfigurationError

MARKER = 'robotic-suite-env.json'


def clean_environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONUSERBASE'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def execute(command, *, env=None, cwd=None, capture=False):
    try:
        return subprocess.run([str(x) for x in command], env=env or clean_environment(),
                              cwd=cwd, check=True, text=True, capture_output=capture)
    except subprocess.CalledProcessError as exc:
        raise ConfigurationError(f'command failed (exit {exc.returncode}): {command[0]}\n{exc.stderr or "See command output above"}') from exc


def python_path(path):
    # Do not resolve the executable symlink: that would bypass the venv.
    return Path(path).absolute()/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def save(path, data):
    target = Path(path)/MARKER
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(target)


def managed(path, ready=True):
    path = Path(path).absolute()
    try:
        data = json.loads((path/MARKER).read_text())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'not an rsctl-managed environment: {path}; use env create') from exc
    if data.get('version') != 1 or data.get('path') != str(path):
        raise ConfigurationError('environment metadata is incompatible or environment was moved; recreate it on this target')
    if not python_path(path).is_file():
        raise ConfigurationError('venv Python is missing; recreate the environment')
    if ready and data.get('state') != 'ready':
        raise ConfigurationError('environment creation is incomplete; inspect its logs and create a new environment')
    return path, data


def freeze(path):
    result = execute([python_path(path), '-I', '-m', 'pip', 'freeze', '--all'], capture=True)
    (Path(path)/'requirements.lock.txt').write_text(result.stdout)


def create(path, platform, sdk, *, python=sys.executable, requirements=None, mock=False):
    from .cli import load_profile, ASSETS
    profile = load_profile(platform)
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise ConfigurationError(f'refusing to overwrite existing environment/path: {path}')
    sdk = Path(sdk).resolve()
    if not sdk.exists():
        raise ConfigurationError(f'SDK source directory or wheel does not exist: {sdk}')
    recipe = Path(requirements).resolve() if requirements else ASSETS/'deployment/requirements'/('mock.txt' if mock else profile['python_environment']['requirements'])
    if not recipe.is_file():
        raise ConfigurationError(f'requirements file missing: {recipe}')
    info = json.loads(execute([python, '-I', '-c', 'import sys,json; print(json.dumps(list(sys.version_info[:2])))'], capture=True).stdout)
    if tuple(info) < tuple(map(int, profile['python_min'].split('.'))):
        raise ConfigurationError(f'Python >= {profile["python_min"]} required')
    execute([python, '-I', '-m', 'venv', str(path)])
    (path/'COLCON_IGNORE').touch()
    data = {'version':1, 'path':str(path), 'platform':platform, 'mock':mock,
            'state':'creating', 'python_version':info, 'sdk_source':str(sdk),
            'requirements_source':str(recipe), 'applications':{}}
    save(path, data)
    # Keep original recipe provenance; pip resolves includes relative to the original file.
    (path/'requirements.requested.txt').write_text(recipe.read_text())
    execute([python_path(path), '-I', '-m', 'pip', 'install', str(sdk), '-r', str(recipe)])
    execute([python_path(path), '-I', '-m', 'pip', 'check'])
    freeze(path)
    data['state'] = 'ready'
    save(path, data)
    return path


def check(path):
    from .cli import load_profile
    path, data = managed(path)
    profile = load_profile(data['platform'])
    probe = "import json,sys,platform; print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,python=platform.python_version(),architecture=platform.machine())))"
    result = json.loads(execute([python_path(path), '-I', '-c', probe], capture=True).stdout)
    cfg = (path/'pyvenv.cfg').read_text().lower()
    isolated = result['prefix'] != result['base_prefix'] and 'include-system-site-packages = false' in cfg
    results = [{'check':'venv isolation', 'status':'passed' if isolated else 'failed',
                'detail':f"Python {result['python']} at {python_path(path)}; inherited PYTHONPATH removed"}]
    arch_ok = result['architecture'] in profile['architectures']
    results.append({'check':'target architecture', 'status':'not verifiable' if data['mock'] else ('passed' if arch_ok else 'failed'),
                    'detail':f"host={result['architecture']}; target={profile['architectures']}; mock does not certify target"})
    results.append({'check':'runtime', 'status':'not verifiable',
                    'detail':'Run the selected application pipeline check in this environment; no suite backend probe'})
    try:
        execute([python_path(path), '-I', '-m', 'pip', 'check'], capture=True)
        results.append({'check':'Python dependencies','status':'passed','detail':'pip check succeeded'})
    except ConfigurationError as exc:
        results.append({'check':'Python dependencies','status':'failed','detail':str(exc)})
    return results


def app_name(name):
    import keyword
    if not re.fullmatch(r'[a-z][a-z0-9_]*',name) or keyword.iskeyword(name):
        raise ConfigurationError('application must be a valid lowercase Python package name')


def install(name, path, source=None, ros_workspace=None):
    app_name(name)
    path, data = managed(path)
    if data['applications'] and name not in data['applications']:
        raise ConfigurationError('one application per deployment venv; create another environment')
    python_paths = []
    if ros_workspace:
        # colcon is a separate system tool: never execute it through the AI venv.
        workspace = Path(ros_workspace).resolve()
        import xml.etree.ElementTree as ET
        matches = []
        for xml in (workspace/'src').rglob('package.xml'):
            try:
                if ET.parse(xml).getroot().findtext('name') == name:
                    matches.append(xml)
            except ET.ParseError as exc:
                raise ConfigurationError(f'invalid ROS package metadata: {xml}') from exc
        if len(matches) != 1:
            raise ConfigurationError(f'expected one ROS source package named {name} under {workspace}/src')
        colcon = shutil.which('colcon')
        if not colcon:
            raise ConfigurationError('colcon missing; install/source ROS build tools separately')
        # Preserve sourced ROS build paths, but remove an activated Python venv from PATH.
        build_env = os.environ.copy()
        active = build_env.pop('VIRTUAL_ENV', None)
        if active:
            build_env['PATH'] = os.pathsep.join(p for p in build_env['PATH'].split(os.pathsep) if Path(p).parent != Path(active))
            colcon = shutil.which('colcon',path=build_env['PATH'])
        if not colcon:
            raise ConfigurationError('system colcon not found outside the activated venv')
        execute([colcon, 'build', '--packages-select', name], env=build_env, cwd=workspace)
        major,minor = data['python_version']
        root = workspace/'install'/name
        python_paths = [str(p) for p in root.glob(f'lib/python{major}.{minor}/site-packages')]
        if not python_paths:
            raise ConfigurationError('no matching Python installation from colcon; use a compatible interpreter and an isolated colcon install layout')
    else:
        if not source:
            raise ConfigurationError('specify --source PATH for Python installation or --ros-workspace PATH for colcon')
        source = Path(source).resolve()
        if not source.exists():
            raise ConfigurationError(f'application source missing: {source}')
        execute([python_path(path), '-I', '-m', 'pip', 'install', str(source)])
    execute([python_path(path), '-I', '-m', 'pip', 'check'])
    env = clean_environment()
    if python_paths:
        env['PYTHONPATH'] = os.pathsep.join(python_paths)
    # Inspect module entry without executing the application. No cwd import fallback.
    command = [python_path(path)] + ([] if python_paths else ['-I'])
    execute(command+['-c', 'import importlib.util,sys; assert importlib.util.find_spec(sys.argv[1]+".__main__") is not None, "application needs __main__.py"', name],env=env,cwd=path)
    freeze(path)
    data['applications'][name] = {'python_paths':python_paths, 'source':str(source) if source else None,
                                 'ros_workspace':str(Path(ros_workspace).resolve()) if ros_workspace else None}
    save(path,data)


def run(name, path, arguments):
    app_name(name)
    path, data = managed(path)
    if name not in data['applications']:
        raise ConfigurationError(f'application {name} is not installed in this environment; use rsctl install')
    env = clean_environment()
    paths = data['applications'][name]['python_paths']
    if paths:
        if any(not Path(p).is_dir() for p in paths):
            raise ConfigurationError('colcon install path missing; rebuild and install application again')
        env['PYTHONPATH'] = os.pathsep.join(paths)
    env['VIRTUAL_ENV'] = str(path)
    env['PATH'] = str(python_path(path).parent)+os.pathsep+env.get('PATH','')
    # -I ignores cwd/user site; colcon mode adds only explicit package roots via runpy.
    code = 'import runpy,sys,json; sys.path[0:0]=json.loads(sys.argv.pop(1)); name=sys.argv.pop(1); sys.argv[0]=name; runpy.run_module(name,run_name="__main__",alter_sys=True)'
    command = [str(python_path(path)), '-I', '-c', code, json.dumps(paths), name, *arguments]
    return subprocess.run(command,env=env).returncode
