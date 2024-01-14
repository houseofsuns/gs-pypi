#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    pypi_db.py
    ~~~~~~~~~~

    PyPI package database

    :copyright: (c) 2013-2015 by Jauhien Piatlicki
    :license: GPL-2, see LICENSE for more details.
"""

import datetime
import enum
import json
import operator
import os
import pathlib
import re
import string
import subprocess
import tempfile

from g_sorcery.exceptions import DownloadingError
from g_sorcery.fileutils import wget
from g_sorcery.g_collections import (
    Dependency, Package, serializable_elist, Version)
from g_sorcery.package_db import DBGenerator
from g_sorcery.logger import Logger

_logger = Logger()


PYTHON_VERSIONS = {Version((3, 10)), Version((3, 11)), Version((3, 12))}


def containment(fun):
    import functools

    @functools.wraps(fun)
    def newfun(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            import traceback
            _logger.error(traceback.format_exc())
            _logger.error(f'ARGUMENTS {args=} {kwargs=}')

    return newfun


class Operator(enum.Enum):
    LESS = 1
    LESSEQUAL = 2
    EQUAL = 3
    SIMILAR = 4
    GREATEREQUAL = 5
    GREATER = 6
    UNEQUAL = 7

    def compare(self, first, second):
        comparators = {
            Operator.LESS: operator.lt,
            Operator.LESSEQUAL: operator.le,
            Operator.EQUAL: operator.eq,
            Operator.SIMILAR: operator.eq,
            Operator.GREATEREQUAL: operator.ge,
            Operator.GREATER: operator.gt,
            Operator.UNEQUAL: operator.ne,
        }
        return comparators[self](first, second)


def parse_version(s, minlength=0):
    if mo := re.fullmatch(r'([0-9]+[\.0-9]*)(.*)', s.strip()):
        version, tail = mo.groups('0')
        components = tuple(map(int, filter(None, version.split('.'))))
        if tail and tail not in {'*', 'dev'}:
            _logger.warn(f'Omitted version tail `{tail}`.')
        if len(components) < minlength:
            components += (0,) * (minlength - len(components))
        return Version(components)
    else:
        _logger.warn(f'Unparsable version `{s}`.')
        return Version((0,) * max(2, minlength))


def parse_operator(s):
    match s.strip():
        case '<':
            return Operator.LESS
        case '<=':
            return Operator.LESSEQUAL
        case '==':
            return Operator.EQUAL
        case '===':
            return Operator.EQUAL
        case '~=':
            return Operator.SIMILAR
        case '>=':
            return Operator.GREATEREQUAL
        case '>':
            return Operator.GREATER
        case '!=':
            return Operator.UNEQUAL
        case _:
            _logger.warn(f'Unparsable operator `{s}`.')
            return Operator.GREATEREQUAL


def extract_requires_python(requires_python):
    default_py_versions = list(sorted(PYTHON_VERSIONS))

    if not requires_python or not requires_python.strip():
        return default_py_versions

    # clean real world data
    requires_python = requires_python.replace(' ', '')

    req_atoms = list(map(lambda s: s.strip(),
                         requires_python.split(',')))
    req_parsed = []
    for req_atom in req_atoms:
        if mo := re.fullmatch(r'([=<>!~]+)(.+)', req_atom):
            op = parse_operator(mo.groups()[0])
            version = parse_version(mo.groups()[1], minlength=2)
            req_parsed.append((op, version))
        else:
            _logger.warn(f'Unhandled requires_python atom `{req_atom}`!')

    py_versions = list(sorted(PYTHON_VERSIONS))
    for op, version in req_parsed:
        py_versions = [v for v in py_versions if op.compare(v, version)]
    if (not py_versions
        and any(op in {Operator.EQUAL, Operator.SIMILAR}
                for op, _ in req_parsed)):
        # Fix for broken version specs in the wild.
        # Some packages supporting e.g. 3.7 and above wrongly depend on ~=3.7.
        _logger.warn(f'Used default py for boguous spec `{requires_python}`.')
        return default_py_versions
    return py_versions


def requires_python_from_classifiers(classifiers):
    default_py_versions = list(sorted(PYTHON_VERSIONS))
    classifiers = set(classifiers)

    ret = []
    for version in default_py_versions:
        if f"Programming Language :: Python :: {version}" in classifiers:
            ret.append(version)
    return ret


def extract_requires_dist(requires_dist, substitutions):
    ret = []
    if not requires_dist:
        return ret
    for entry in requires_dist:
        if mo := re.fullmatch(
                (r'([-_\.a-zA-Z0-9]+)\s*(\[[-_a-zA-Z0-9,\s]+\])?\s*'
                 r'([(=<>!~][=<>!~0-9a-zA-Z\.(),\s\*]+)?\s*(;.*)?'),
                entry.strip()):
            name, _, versionbounds, conditions = mo.groups()
            # We ignore the extra in the dependency spec to avoid collisions
            # (or better lack of hits) with the main tree

            dep = {
                'name': filter_package_name(name, substitutions),
                'versionbound': None,
                'extras': [],
            }

            if versionbounds and versionbounds.strip():
                opranking = [
                    Operator.EQUAL,
                    Operator.SIMILAR,
                    Operator.LESS,
                    Operator.LESSEQUAL,
                    Operator.GREATER,
                    Operator.GREATEREQUAL,
                    Operator.UNEQUAL,
                ]
                cleaned = ''.join(c for c in versionbounds
                                  if c not in ' ()')
                topop, topversion = None, None
                for part in cleaned.split(','):
                    if mobj := re.fullmatch(r'([=<>!~]+)(.+)', part.strip()):
                        # FIXME in the case `==2.6.*` we may want to relax the
                        # resulting dependency to `<=2.7` or `>=2.6` instead
                        # of using `==2.6.0`
                        op = parse_operator(mobj.groups()[0])
                        version = parse_version(mobj.groups()[1], minlength=2)
                        if topop is None or (opranking.index(op)
                                             < opranking.index(topop)):
                            topop, topversion = op, version
                    else:
                        _logger.warn(f'Unhandled version bound `{part}`.')
                if topop:
                    opencoding = {
                        Operator.EQUAL: '~',
                        Operator.SIMILAR: '~',
                        Operator.LESS: '<',
                        Operator.LESSEQUAL: '<=',
                        Operator.GREATER: '>',
                        Operator.GREATEREQUAL: '>=',
                        Operator.UNEQUAL: '>',
                    }
                    dep['versionbound'] = (opencoding[topop], topversion)

            skip = False
            if conditions:
                cleaned = ''.join(c for c in conditions if c not in '();')
                terms = [term.strip()
                         for clause in cleaned.split(' and ')
                         for term in clause.split(' or ')]
                for term in terms:
                    term = term.replace(' ', '')
                    if mobj := re.fullmatch(
                            r'''extra\s*==\s*['"]([-_\.a-zA-Z0-9]+)['"]''',
                            term):
                        dep['extras'].append(
                            sanitize_useflag(mobj.groups()[0]))
                    elif (term.startswith('platform_python_implementation')
                          or term.startswith('implementation_name')):
                        # FIXME handle python implementation differences
                        pass
                    elif (term.startswith('python_version')
                          or term.startswith('python_full_version')):
                        op = parse_operator(''.join(
                            c for c in term if c in '=<>!~'))
                        version = parse_version(''.join(
                            c for c in term if c in '0123456789.'))
                        if not any(op.compare(available, version)
                                   for available in PYTHON_VERSIONS):
                            skip = True
                        # FIXME if only some versions match it would be nice
                        # to make this into a conditional dependency like so:
                        # $(python_gen_cond_dep 'dev-python/tomli' 3.{9..10})
                    elif ((term.startswith('os_name')
                           or term.startswith('platform_system')
                           or term.startswith('sys_platform'))
                          and (('windows' in term.lower()
                                or 'nt' in term.lower()
                                or 'win32' in term.lower())
                               and '==' in term)):
                        skip = True
                    elif (term.startswith('os_name')
                          or term.startswith('platform_system')
                          or term.startswith('sys_platform')):
                        # FIXME handle platform differences
                        pass
                    elif term.startswith('platform_machine'):
                        # FIXME handle architecture differences
                        pass
                    else:
                        # FIXME handle more
                        _logger.warn(f'Ignoring dependency'
                                     f' condition `{term}`.')
            if not skip:
                ret.append(dep)
        else:
            _logger.warn(f'Dropping unexpected dependency `{entry}`.')
    return ret


def filter_package_name(package, substitutions):
    if replacement := substitutions.get(package):
        # Perform some fixed substitutions, primarily to converge with naming
        # in the main tree
        return replacement
    return sanitize_package_name(package)


def sanitize_package_name(package):
    ret = DBGenerator.filter_characters(package.replace('.', '-'), [
            ('a', 'z'), ('A', 'Z'), ('0', '9'), '+_-'])
    if '-' in ret:
        # Fixup invalid package name due to suffix that looks like a version.
        # Note that captial letters seem to be allow by PMS but are forbidden
        # by pkgcore, so we play safe.
        parts = ret.split('-')
        if len(parts) > 1 and re.fullmatch(r'([0-9\.]+)[a-zA-Z]?', parts[-1]):
            ret = '-'.join(parts[:-1]) + '_' + parts[-1]
    # guarantee that the package name starts with a letter or number
    if not re.match(r'^[a-zA-Z0-9]', ret):
        ret = 'x' + ret
    return ret


def sanitize_useflag(useflag):
    ret = DBGenerator.filter_characters(useflag.replace('.', '-'), [
            ('a', 'z'), ('A', 'Z'), ('0', '9'), '+_-@'])
    # guarantee that the useflag starts with a letter or number
    if not re.match(r'^[a-zA-Z0-9]', ret):
        ret = 'x' + ret
    return ret


class PypiDBGenerator(DBGenerator):
    """
    Implementation of database generator for PYPI backend.
    """

    def generate_tree(self, pkg_db, common_config, config):
        self.exclude = set(self.combine_config_lists(
            [common_config, config], 'exclude'))
        self.wanted = set(self.combine_config_lists(
            [common_config, config], 'wanted'))
        self.substitutions = self.combine_config_dicts(
            [common_config, config], 'substitute')
        self.nonice = set(self.combine_config_lists(
            [common_config, config], 'nonice'))
        self.mainpkgs = self.lookupmaintree(common_config, config)
        # Now proceed with normal flow
        super().generate_tree(pkg_db, common_config, config)

    def lookupmaintree(self, common_config, config):
        ret = set()
        fname = "dev-python.html"
        pattern = (
            r'<a[^>]*/gentoo.git/tree/dev-python[^>]*>([-a-zA-Z0-9\._]+)</a')
        with tempfile.TemporaryDirectory() as download_dir:
            if wget(config['gentoo_main_uri'], download_dir, fname):
                raise DownloadingError("Retrieving main tree directory failed")
            with open(pathlib.Path(download_dir) / fname) as htmlfile:
                for line in htmlfile.readlines():
                    if 'd---------' in line:
                        if mo := re.search(pattern, line):
                            ret.add(mo.group(1))
        _logger.info(f'Total of main tree packages: {len(ret)}.')
        return ret

    def get_download_uries(self, common_config, config):
        """
        Get URI of packages index.
        """
        _logger.info('Retrieving package index.')
        return [{"uri": config["data_uri"], "open_file": True}]

    def parse_datum(self, datapath):
        package = datapath.stem
        if package in self.exclude:
            return {}
        if (not os.environ.get('GSPYPI_INCLUDE_UNCOMMON')
                and package not in self.wanted):
            # we only include a selected set of packages as otherwise the
            # overlay becomes unwieldy
            return {}
        self.check_confusion(datapath)
        with open(datapath, 'r') as datafile:
            return {package: json.load(datafile)}

    def check_confusion(self, entry):
        name = entry.stem
        checks = []
        if '-' in name:
            checks.append(('-', '_'))
        if '_' in name:
            checks.append(('_', '-'))
        for check in checks:
            newname = name.replace(*check)
            candidate = entry.parent / f'{newname}{entry.suffix}'
            if candidate.exists():
                _logger.warn(f'Possible hyphen-confusion: {name} and'
                             f' {newname}')

    def parse_data(self, data_f):
        """
        Parse package data.
        """
        data = {}
        zipfile = pathlib.Path(data_f.name)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = pathlib.Path(tmpdirname)
            subprocess.run(['unzip', str(zipfile), '-d', str(tmpdir)],
                           stdout=subprocess.DEVNULL, check=True)
            datadir = tmpdir / 'pypi-json-data-main' / 'release_data'
            for firstletterdir in datadir.iterdir():
                # There exist some metadata files which do not interest us
                if firstletterdir.is_dir():
                    for second in firstletterdir.iterdir():
                        if second.is_dir():
                            for entry in second.iterdir():
                                if entry.is_file() and entry.suffix == '.json':
                                    data.update(self.parse_datum(entry))
                        elif second.is_file() and second.suffix == '.json':
                            # Some entries are on the first level
                            data.update(self.parse_datum(second))
        return data

    @staticmethod
    def name_output(package, filtered_package):
        ret = package
        if package != filtered_package:
            ret += f" (as {filtered_package})"
        return ret

    def maybe_add_package(self, pkg_db, package, data):
        nout = self.name_output(data['realname'], package.name)
        if pkg_db.in_category(package.category, package.name):
            _logger.warn(f"Rejected package {nout} for collision.")
            return False
        pkg_db.add_package(package, data)
        return True

    def process_data(self, pkg_db, data, common_config, config):
        """
        Process parsed package data.
        """
        category = "dev-python"
        pkg_db.add_category(category)

        common_data = {}
        common_data["eclasses"] = ['g-sorcery', 'gs-pypi']
        common_data["maintainer"] = [{'email': 'gentoo@houseofsuns.org',
                                      'name': 'Markus Walter'}]
        pkg_db.set_common_data(category, common_data)

        for package, pkg_data in data['main.zip'].items():
            self.process_datum(pkg_db, common_config, config, package,
                               pkg_data)

    @containment
    def process_datum(self, pkg_db, common_config, config, package, pkg_data):
        """
        Process one parsed package datum.
        """
        _logger.info(f'Processing {package}.')
        category = "dev-python"
        aberrations = []

        pkg_datum = None
        mtime = None
        src_uri = None
        best_ver = None
        for v, datum in pkg_data.items():
            cur_ver = parse_version(datum['info']['version'])
            for distmeta in datum['urls']:
                if distmeta['packagetype'] == 'sdist':
                    currentdate = datetime.datetime.fromisoformat(
                        distmeta['upload_time_iso_8601'])
                    if mtime is None or mtime < currentdate:
                        mtime = currentdate
                        pkg_datum = datum
                        src_uri = distmeta['url']
                        best_ver = cur_ver
            if best_ver and cur_ver > best_ver:
                _logger.warn(f'Dropped better version `{cur_ver}`.')
        if not src_uri:
            _logger.warn(f'No source distfile for {package} -- dropping.')
            return

        version = pkg_datum['info']['version']
        top_version = max(map(parse_version, pkg_data))
        if top_version > parse_version(version):
            aberrations.append(f"topver {top_version}")
        homepage = pkg_datum['info']['home_page'] or ""
        if not homepage:
            purls = pkg_datum['info'].get('project_urls') or {}
            for key in ["Homepage", "homepage"]:
                homepage = purls.get(key, "")
                if homepage:
                    break
        homepage = self.escape_bash_string(self.strip_characters(homepage))

        pkg_license = pkg_datum['info']['license'] or ''
        # This has to avoid any characters that have a special meaning for
        # dependency specification, these are: !?|^()
        pkg_license = self.filter_characters(
            (pkg_license.splitlines() or [''])[0],
            mask_spec=[
                ('a', 'z'), ('A', 'Z'), ('0', '9'),
                ''' #%'*+,-./:;=<>&@[]_{}~'''])
        pkg_license = self.convert([common_config, config], "licenses",
                                   pkg_license)
        pkg_license = self.escape_bash_string(pkg_license)

        requires_python = extract_requires_python(
            pkg_datum['info']['requires_python'])
        for addon in requires_python_from_classifiers(
                pkg_datum['info'].get('classifiers', [])):
            if addon not in requires_python:
                requires_python.append(addon)
        if not requires_python:
            _logger.warn(f'No valid python versions for {package}'
                         f' -- dropping.')
            return
        py_versions = list(map(
            lambda version: f'{version.components[0]}_{version.components[1]}',
            requires_python))

        if len(py_versions) == 1:
            python_compat = '( python' + py_versions[0] + ' )'
        else:
            python_compat = '( python{' + (','.join(py_versions)) + '} )'

        requires_dist = extract_requires_dist(
            pkg_datum['info']['requires_dist'], self.substitutions)

        dependencies = []
        useflags = set()
        for dep in requires_dist:
            for extra in (dep['extras'] or [""]):
                if (dep['name'] in self.mainpkgs) and dep["versionbound"]:
                    # keep version bounds for packages in the main tree as
                    # there will probably be some choice in the relevant cases
                    dop, dver = dep["versionbound"]
                else:
                    # ignore version bound as we only provide the most recent
                    # version anyway so there is no choice. Additionally this
                    # fixes broken dependency specs where there either is an
                    # error or which are simply outdated.
                    dop, dver = "", ""
                dependencies.append(Dependency(
                    category, dep['name'], usedep='${PYTHON_USEDEP}',
                    useflag=extra, version=str(dver), operator=str(dop)))
                if extra:
                    useflags.add(extra)

        filtered_package = filter_package_name(package, self.substitutions)
        # Some packages have uppercase letters in their names that are
        # normalized in much of the PyPI pipeline but exposed in others
        literal_package = pkg_datum['info']['name']
        if literal_package.lower() != package:
            _logger.warn(
                f'Unexpected package name {literal_package} for {package}.')

        filtered_version = version
        version_filters = [(r'^(.*[0-9]+)\.?a([0-9]+)$', r'\1_alpha\2'),
                           (r'^(.*[0-9]+)\.?b([0-9]+)$', r'\1_beta\2'),
                           (r'^(.*[0-9]+)\.?post([0-9]+)$', r'\1_p\2'),
                           (r'^(.*[0-9]+)\.?rc([0-9]+)$', r'\1_rc\2'),
                           (r'^(.*[0-9]+)\.?dev([0-9]+)$', r'\1_pre\2')]
        for pattern, replacement in version_filters:
            # FIXME convert more versions to acceptable versions
            filtered_version = re.sub(pattern, replacement, filtered_version)
        if not re.fullmatch(r'[0-9]+(\.[0-9]+)*[a-z]?'
                            r'(_(alpha|beta|rc|p|pre)[0-9]+)?',
                            filtered_version):
            bad_version = filtered_version
            filtered_version = "%04d%02d%02d" % (
                mtime.year, mtime.month, mtime.day)
            _logger.warn(f'Version {bad_version} is bad'
                         f' using {filtered_version}.')
            aberrations.append(f"badver {bad_version}")

        nice_src_uri = src_uri
        filename = src_uri.split('/')[-1]
        pattern = (r'https://files\.pythonhosted\.org/packages'
                   r'/[0-9a-f]+/[0-9a-f]+/[0-9a-f]+/.*')
        if re.fullmatch(pattern, src_uri.lower()):
            filepath = src_uri.removesuffix(filename)
            suffix = ''
            for extension in ['.tar.gz', '.tar.bz2', '.zip']:
                if filename.endswith(extension):
                    suffix = extension
                    filename = filename.removesuffix(extension)
                    break
            src_uri_filters = [
                (f'{version}', '${REALVERSION}'),
                (f'{package}', '${REALNAME}'),
                (f'{literal_package}', '${LITERALNAME}'),
                (f'{package.replace("-", "_")}', '${REALNAME//-/_}'),
                (f'{package.replace("_", "-")}', '${REALNAME//_/-}'),
                (f'{literal_package.replace("-", "_")}',
                 '${LITERALNAME//-/_}'),
                (f'{literal_package.replace("_", "-")}',
                 '${LITERALNAME//_/-}'),
            ]
            for pattern, replacement in src_uri_filters:
                filename = filename.replace(pattern, replacement)
            filename = filename + suffix
            npattern = r'\$\{(LITERALNAME|REALNAME)[-_/]*\}-\$\{REALVERSION\}'
            if ((mo := re.match(npattern, filename))
                    and package[0] in string.ascii_letters + string.digits
                    and package not in self.nonice):
                name = mo.group(1)
                # Use redirect URL to avoid churn through the embedded hashes
                # in the actual URL
                nice_src_uri = (
                    f'https://files.pythonhosted.org/packages/source'
                    f'/${{{name}::1}}/${{{name}}}/{filename}')
            else:
                _logger.warn(f'Unsubstituted SRC_URI `{src_uri}`.')
                nice_src_uri = filepath + filename
        else:
            _logger.warn(f'Unexpected SRC_URI `{src_uri}`.')

        description = pkg_datum['info']['summary'] or ''
        if aberrations:
            description += " [" + ", ".join(aberrations) + "]"
        filtered_description = self.escape_bash_string(self.strip_characters(
            description))

        ebuild_data = {}
        ebuild_data["realname"] = (
            "${PN}" if package == filtered_package else package)
        ebuild_data["literalname"] = (
            "${PN}" if filtered_package == literal_package
            else literal_package)
        ebuild_data["realversion"] = (
            "${PV}" if version == filtered_version else version)
        ebuild_data["mtime"] = mtime.isoformat()

        ebuild_data["description"] = filtered_description

        ebuild_data["homepage"] = homepage
        ebuild_data["license"] = pkg_license
        ebuild_data["src_uri"] = nice_src_uri
        ebuild_data["sourcefile"] = filename
        ebuild_data["repo_uri"] = nice_src_uri.removesuffix(
            ebuild_data["sourcefile"])
        ebuild_data["python_compat"] = python_compat
        ebuild_data["iuse"] = " ".join(sorted(useflags))
        deplist = serializable_elist(separator="\n\t")
        deplist.extend(dependencies)
        ebuild_data["dependencies"] = deplist

        self.maybe_add_package(
                pkg_db,
                Package(category, filtered_package, filtered_version),
                ebuild_data)

    def convert_internal_dependency(self, configs, dependency):
        """
        At the moment we have only internal dependencies, each of them
        is just a package name.
        """
        return Dependency("dev-python", dependency)
