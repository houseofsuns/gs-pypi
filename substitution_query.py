#!/usr/bin/env python3

import concurrent.futures
import json
import pathlib
import re

import requests


def pypi_normalize(pkg):
    return pkg.lower().replace('_', '-').replace('.', '-')


def lookupmaintree():
    ret = {}
    url = "https://gitweb.gentoo.org/repo/gentoo.git/tree/dev-python"
    pattern = (
        r'<a[^>]*/gentoo.git/tree/dev-python[^>]*>([-a-zA-Z0-9\._]+)</a')
    with requests.get(url) as req:
        for line in req.text.splitlines():
            if 'd---------' in line:
                if mo := re.search(pattern, line):
                    pkg = mo.group(1)
                    ret[pypi_normalize(pkg)] = pkg
    return ret


def load_pkg(pkg):
    with requests.get(f'https://pypi.org/pypi/{pkg}/json') as req:
        data = req.json()
        return data['info']['name']


catalog = None
local_file = pathlib.Path('./local_authority.json')
if local_file.exists():
    with open(local_file) as f:
        catalog = json.load(f)


def load_pkg_local(pkg):
    return catalog.get(pkg, pkg)


with open('./gs-pypi.json') as f:
    config = json.load(f)

old_substitutions = config['common_config']['substitute']
new_substitutions = {}
maintree = lookupmaintree()

print('Starting process.')
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Start the load operations and mark each future with its URL
    futures = {executor.submit(load_pkg, pkg): pkg
               for pkg in config['common_config']['wanted']}
    for future in concurrent.futures.as_completed(futures):
        pkg = futures[future]
        if pkg != pypi_normalize(pkg):
            print(f'Unnormalized package {pkg} in wanted list!')
        try:
            realname = future.result()
        except Exception as exc:
            print(f'{pkg} generated an exception: {exc}')
        else:
            chosen = maintree.get(pkg, realname)
            if pkg not in old_substitutions:
                if pkg != chosen:
                    new_substitutions[pkg] = chosen
                    if pkg in maintree:
                        if chosen == realname:
                            print(f'Added main {pkg} => {chosen}')
                        else:
                            print(f'Preferred main {pkg} => {chosen}'
                                  f' (ignoring {realname})')
                    else:
                        print(f'Added {pkg} => {chosen}')
                elif pkg == chosen != realname:
                    print(f'Overruled main {pkg} => {chosen}'
                          f' (ignoring {realname})')
            else:
                if old_substitutions[pkg] != chosen:
                    print(f'Kept {pkg} => {old_substitutions[pkg]}'
                          f' (over new {chosen})')

print('Dumping result.')
with open('./substitutions.json', 'w') as f:
    json.dump(dict(sorted(new_substitutions.items())), f)
print('Done.')
