import argparse
import importlib.util
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys
import tempfile
from robotic_inference import run_pipeline
from robotic_inference.configuration import read_yaml
from robotic_inference.errors import ConfigurationError

ASSETS = Path(str(files('rsctl').joinpath('assets')))
PROFILE_FILES = {'nvidia':'nvidia_jetson', 'intel':'intel_x86', 'qualcomm':'qualcomm_qcs'}

def load_profile(name):
    if name not in PROFILE_FILES:
        raise ConfigurationError(f'unknown platform {name!r}; choose nvidia, intel, qualcomm')
    p = read_yaml(ASSETS/'deployment/profiles'/f'{PROFILE_FILES[name]}.yaml')
    required = {'name','architectures','runtime','requirements','environment','devices','os','python_min','ros_distros','dockerfile','python_environment'}
    if set(p) != required or p['name'] != name:
        raise ConfigurationError(f'invalid platform profile: {name}')
    for key in ('architectures','requirements','environment','devices','ros_distros'):
        if not isinstance(p[key],list) or any(not isinstance(x,str) for x in p[key]):
            raise ConfigurationError(f'profile {key} must be a string list')
    if p['runtime'] not in ('tensorrt','openvino','qnn') or not re.fullmatch(r'\d+\.\d+',p['python_min']):
        raise ConfigurationError('invalid runtime or python_min in profile')
    if not (ASSETS/'deployment/docker'/p['dockerfile']).is_file():
        raise ConfigurationError('profile Dockerfile missing')
    settings = p['python_environment']
    if not isinstance(settings, dict) or settings.get('system_site_packages') is not False:
        raise ConfigurationError('platform must use an isolated application venv')
    if not isinstance(settings.get('requirements'), str) or not (ASSETS/'deployment/requirements'/settings['requirements']).is_file():
        raise ConfigurationError('platform requirements recipe missing')
    return p

def init_app(name, destination, template):
    if template != 'ros2_perception_demo':
        raise ConfigurationError('unknown template; available: ros2_perception_demo')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in ('test','setup'):
        raise ConfigurationError('package name must start with lowercase letter and contain lowercase letters, digits or underscores')
    import keyword
    if keyword.iskeyword(name):
        raise ConfigurationError('package name cannot be a Python keyword')
    target = Path(destination).resolve()/name
    if target.exists():
        raise ConfigurationError(f'refusing to overwrite existing path: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    source = ASSETS/'templates'/template
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staging = Path(temp)/name
        for path in source.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts:
                relative = str(path.relative_to(source)).replace(template,name)
                dest = staging/relative
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(path.read_text().replace(template,name))
        staging.rename(target)
    return target

def doctor(name):
    p = load_profile(name)
    results = []
    def add(check,status,detail): results.append({'check':check,'status':status,'detail':detail})
    add('OS','passed' if platform.system()==p['os'] else 'failed', platform.system())
    add('architecture','passed' if platform.machine() in p['architectures'] else 'failed',platform.machine())
    minimum = tuple(map(int,p['python_min'].split('.')))
    add('Python','passed' if sys.version_info[:2]>=minimum else 'failed',platform.python_version())
    distro = os.environ.get('ROS_DISTRO')
    if not distro:
        add('ROS 2','not verifiable','ROS_DISTRO unset; install/source Humble or Jazzy (installation cannot be inferred from an unsourced shell)')
    else:
        add('ROS distro','passed' if distro in p['ros_distros'] else 'failed',distro)
        try:
            import rclpy
            add('rclpy','passed','Python import succeeded; ROS graph not tested')
        except (ImportError,OSError) as e:
            add('rclpy','not installed',str(e))
    # Environment hints only; the selected pipeline owns real runtime/model checks.
    runtime_module = {'nvidia':'torch_tensorrt','intel':'openvino','qualcomm':None}[name]
    present_runtime = bool(runtime_module and importlib.util.find_spec(runtime_module))
    add('inference runtime', 'not verifiable',
        f'Python runtime module present: {present_runtime}; use validate --config for the platform pipeline')
    add('Docker','passed' if shutil.which('docker') else 'not installed','CLI only; daemon and runtime access not verified')
    present = [d for d in p['devices'] if Path(d).exists()]
    add('accelerator access','not verifiable',f'device nodes found: {present}; execute a target smoke test; Intel CPU does not require accelerator nodes')
    for alternatives in p['environment']:
        found = [key for key in alternatives.split('|') if os.environ.get(key) and Path(os.environ[key]).is_dir()]
        add(alternatives,'passed' if found else 'not installed',f'valid directories: {found}; directory presence does not validate ABI')
    if name == 'nvidia':
        add('JetPack/L4T','not verifiable','Check board BSP, CUDA/TensorRT compatibility and NVIDIA container runtime on target')
    if name == 'qualcomm':
        add('pipeline implementation','not verifiable','Qualcomm execution is owned by the selected application pipeline')
    return results

def validate(config, runtime=False):
    result=run_pipeline(config, check=True, require_runtime=runtime, capture=True)
    if result.returncode:
        raise ConfigurationError(result.stderr.strip() or result.stdout.strip() or f'Pipeline check exited {result.returncode}')
    return result.stdout

def build_plan(name, app, context, base_image):
    p = load_profile(name)
    if not re.fullmatch(r'[a-z][a-z0-9_]*',app):
        raise ConfigurationError('app must be a lowercase ROS package name')
    context = Path(context).resolve()
    missing = []
    if not (context/'packages/robotic_inference_sdk/pyproject.toml').is_file():
        missing.append('context/packages/robotic_inference_sdk (copy SDK source into build context)')
    xml = context/'application/package.xml'
    if not xml.is_file():
        missing.append('context/application/package.xml (copy generated ROS application into build context)')
    else:
        import xml.etree.ElementTree as ET
        try:
            if ET.parse(xml).getroot().findtext('name') != app:
                raise ConfigurationError('--app must match application/package.xml name')
        except ET.ParseError as e:
            raise ConfigurationError(f'invalid package.xml: {e}') from e
    if not base_image:
        missing.append('explicit validated target BASE_IMAGE')
    command = ['docker','build','--file',str(ASSETS/'deployment/docker'/p['dockerfile']),
               '--build-arg',f'BASE_IMAGE={base_image or "<TARGET_ROS_BASE_IMAGE>"}',
               '--build-arg',f'APP_NAME={app}','--tag',f'{app}:{name}',str(context)]
    return {'mode':'DRY RUN ONLY — image/device compatibility is unverified',
            'command':shlex.join(command), 'requirements':p['requirements'], 'missing':missing,
            'execution_supported':False}

def main(argv=None):
    parser = argparse.ArgumentParser(prog='rsctl',description='Standalone app generation and deployment diagnostics')
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('init'); p.add_argument('--name',required=True); p.add_argument('--template',default='ros2_perception_demo'); p.add_argument('--destination',default='.')
    p = sub.add_parser('pipeline-run'); p.add_argument('--config', required=True)
    p = sub.add_parser('doctor'); p.add_argument('--platform',required=True); p.add_argument('--json',action='store_true')
    p = sub.add_parser('validate'); p.add_argument('--config',required=True); p.add_argument('--require-runtime',action='store_true')
    p = sub.add_parser('build'); p.add_argument('--platform',required=True); p.add_argument('--app',required=True); p.add_argument('--dry-run',action='store_true'); p.add_argument('--context',default='.'); p.add_argument('--base-image')
    p = sub.add_parser('env')
    env_sub = p.add_subparsers(dest='env_command', required=True)
    q = env_sub.add_parser('create')
    q.add_argument('--platform', required=True); q.add_argument('--path', default='.venv')
    q.add_argument('--python', default=sys.executable); q.add_argument('--sdk', required=True)
    q.add_argument('--requirements'); q.add_argument('--mock', action='store_true')
    q = env_sub.add_parser('check'); q.add_argument('--path', default='.venv')
    p = sub.add_parser('install'); p.add_argument('app'); p.add_argument('--venv', default='.venv')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source'); group.add_argument('--ros-workspace')
    p = sub.add_parser('run'); p.add_argument('app'); p.add_argument('--venv', default='.venv')
    # Application flags are passed after --, without interpreting them as rsctl options.
    raw = list(sys.argv[1:] if argv is None else argv)
    arguments = []
    if raw and raw[0] in ('run','pipeline-run') and '--' in raw:
        split = raw.index('--'); arguments = raw[split+1:]; raw = raw[:split]
    args = parser.parse_args(raw)
    try:
        if args.command in ('env', 'install', 'run'):
            from . import environments
            if args.command == 'env':
                if args.env_command == 'create':
                    path = environments.create(args.path, args.platform, args.sdk, python=args.python,
                                               requirements=args.requirements, mock=args.mock)
                    print(f'Environment created: {path}. Runtime readiness is reported separately:')
                    result = environments.check(path)
                else:
                    result = environments.check(args.path)
                print(json.dumps(result, indent=2))
                return 1 if any(x['status'] == 'failed' for x in result) else 0
            if args.command == 'install':
                environments.install(args.app,args.venv,args.source,args.ros_workspace)
                print(f'Installed {args.app} into deployment environment {args.venv}')
            else:
                return environments.run(args.app,args.venv,arguments)
        elif args.command == 'init':
            print(init_app(args.name,args.destination,args.template))
        elif args.command == 'doctor':
            result = doctor(args.platform)
            if args.json: print(json.dumps(result,indent=2))
            else:
                for item in result: print(f"[{item['status']}] {item['check']}: {item['detail']}")
            return 1 if any(r['status'] in ('failed','not installed') for r in result) else 0
        elif args.command == 'pipeline-run':
            return run_pipeline(args.config, arguments).returncode
        elif args.command == 'validate':
            print(validate(args.config,args.require_runtime),end='')
            print('Configuration valid; checks performed by the platform pipeline')
        else:
            if not args.dry_run:
                raise ConfigurationError('Real builds are not enabled: use --dry-run; validate the base image, target dependencies and Dockerfile first')
            print(json.dumps(build_plan(args.platform,args.app,args.context,args.base_image),indent=2))
        return 0
    except (ConfigurationError,OSError,ValueError) as e:
        print(f'rsctl: {e}',file=sys.stderr)
        return 2
