#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    pypi_db.py
    ~~~~~~~~~~

    PyPI package database

    :copyright: (c) 2013-2015 by Jauhien Piatlicki
    :license: GPL-2, see LICENSE for more details.
"""

import collections
import concurrent.futures
import copy
import datetime
import enum
import json
import operator
import os
import pprint
import random
import re
import string
import time

import bs4

from g_sorcery.exceptions import DBLayoutError, DownloadingError
from g_sorcery.g_collections import (
    Dependency, Package, serializable_elist, Version)
from g_sorcery.package_db import DBGenerator
from g_sorcery.logger import Logger

_logger = Logger()


def containment(fun):
    import functools

    @functools.wraps(fun)
    def newfun(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
            print('ARGUMENTS', args, kwargs)
            import pathlib
            p = pathlib.Path.home() / 'gs-pypi-tracebacks/'
            p.mkdir(parents=True, exist_ok=True)
            num = len(list(p.glob('*')))
            with open(p / f'tb-{num:05}.txt', 'w') as f:
                traceback.print_exc(file=f)
                f.write('ARGUMENTS\n')
                pprint.pprint(args, f)
                pprint.pprint(kwargs, f)

    return newfun


def print_progress_dot(fun):
    import functools

    @functools.wraps(fun)
    def newfun(*args, **kwargs):
        ret = fun(*args, **kwargs)
        print('.', end='', flush=True)
        return ret

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


class Action(enum.Enum):
    ADD = 1
    UPDATE = 2
    SKIP = 3
    IGNORE = 4
    EXCLUDE = 5
    SKIMP = 6

    def actionable(self):
        return self in {Action.ADD, Action.UPDATE}


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
    py_available = {Version((3, i)) for i in range(9, 12)}
    default_py_versions = list(map(
        Version, [(3, 9), (3, 10), (3, 11)]))
    minimal_py = min(default_py_versions)
    py_versions = []

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
    lower = max((version for op, version in req_parsed
                 if op in {Operator.GREATER, Operator.GREATEREQUAL}),
                default=minimal_py)
    major, minor, *_ = lower.components
    py_versions = (
        [Version((major, j)) for j in range(minor, 99)]
        + [Version((i, j)) for i in range(major+1, 6) for j in range(99)])
    py_versions = list(sorted(set(py_versions) & py_available))
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
                        # FIXME handle python version differences
                        pass
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
        # First store necessary config for further processing
        ignore = {}
        ignore['vacuous'] = set(self.combine_config_lists(
            [common_config, config], 'vacuous'))
        ignore['nosource'] = set(self.combine_config_lists(
            [common_config, config], 'nosource'))
        ignore['nopython'] = set(self.combine_config_lists(
            [common_config, config], 'nopython'))
        ignore['total'] = (ignore['vacuous'] | ignore['nosource']
                           | ignore['nopython'])
        existing = set(self.old_db_lookup)
        if existing & ignore['total']:
            if hits := existing & ignore['vacuous']:
                _logger.warn(f'Overly broad ignore spec! The following'
                             f' packages are not vacuous: {hits}.')
            if hits := existing & ignore['nosource']:
                _logger.warn(f'Overly broad ignore spec! The following'
                             f' packages have a source dist: {hits}.')
            if hits := existing & ignore['nopython']:
                _logger.warn(f'Overly broad ignore spec! The following'
                             f' packages have valid python: {hits}.')
        self.ignore = ignore
        self.exclude = set(self.combine_config_lists(
            [common_config, config], 'exclude'))
        self.wanted = set(self.combine_config_lists(
            [common_config, config], 'wanted'))
        self.substitutions = self.combine_config_dicts(
            [common_config, config], 'substitute')
        # Now proceed with normal flow
        super().generate_tree(pkg_db, common_config, config)

    def pre_clean_for_generation(self, pkg_db):
        """Store a copy."""
        self.old_db = copy.deepcopy(pkg_db)
        try:
            self.old_db.read()
        except DBLayoutError:
            # No DB exists, so we cannot read it
            pass
        self.old_db_lookup = {
            package.name: package
            for package in self.old_db.list_all_packages()
        }

    def get_download_uries(self, common_config, config):
        """
        Get URI of packages index.
        """
        self.repo_uri = config["repo_uri"]
        _logger.info("Retrieving package list.")
        return [{"uri": self.repo_uri + "/simple/", "output": "packages"}]

    def do_download(self, uri):
        data = {}
        attempts = 0
        while attempts < 7:
            attempts += 1
            try:
                self.process_uri(uri, data)
            except DownloadingError as error:
                _logger.warn(str(error))
                time.sleep(1)
                continue
            break
        return data

    def lookup_previous(self, package):
        gentoo_name = filter_package_name(package, self.substitutions)
        try:
            temp = self.old_db.database['dev-python']['packages'][gentoo_name]
            # temp contains a map version -> ebuild data and should have size 1
            return next(iter(temp.values()))
        except KeyError:
            return {}

    def previous_package_version(self, package):
        """May only be called if previous version exists."""
        gentoo_name = filter_package_name(package, self.substitutions)
        temp = self.old_db.database['dev-python']['packages'][gentoo_name]
        return (gentoo_name, next(iter(temp.keys())))

    def parse_data(self, data_f):
        """
        Download and parse packages index. Then download and parse pages for
        all packages.
        """
        soup = bs4.BeautifulSoup(data_f, 'lxml')
        data = {
            '__index': {},
        }

        _logger.info("Selecting packages for update.")
        pkg_uries = []
        for idx, entry in enumerate(soup.find_all('a')):
            package = entry.string
            pathcomponent = entry['href'].removeprefix(
                '/simple/').removesuffix('/')

            previous = self.lookup_previous(package)
            action = Action.UPDATE if previous else Action.ADD
            reason = ''
            if package in self.exclude:
                action = Action.EXCLUDE
            elif (package in self.ignore['total']
                  and not os.environ.get('GSPYPI_FORCE_IGNORED')):
                action = Action.IGNORE
                if package in self.ignore['vacuous']:
                    reason = 'vacuous'
                elif package in self.ignore['nosource']:
                    reason = 'no sdist available'
                elif package in self.ignore['nopython']:
                    reason = 'no valid python version'
                else:
                    reason = 'unknown'
            elif (not os.environ.get('GSPYPI_INCLUDE_UNCOMMON')
                  and package not in self.wanted):
                # by default we update only the most downloaded packages
                action = Action.SKIMP

            if action.actionable():
                if (not os.environ.get('GSPYPI_FORCE_UPDATE')
                        and (rawmtime := previous.get('mtime'))):
                    mtime = datetime.datetime.fromisoformat(rawmtime)
                    now = datetime.datetime.now(datetime.timezone.utc)
                    delta = (now - mtime).total_seconds()
                    # the logic for skipping a package should have the
                    # following properties:
                    # - a package with an update in the last month
                    #   should not be skipped
                    # - a package with no update for two or more years is
                    #   considered dead and should be skipped with high
                    #   probability
                    # - in between we interpolate somehow
                    one_month = 1*30*24*60*60
                    two_years = 2*365*24*60*60
                    minimal_probability_sqrt = 0.1
                    update_probability_sqrt = max(
                        minimal_probability_sqrt,
                        1 - ((delta - one_month)
                             / (two_years - one_month)))
                    if update_probability_sqrt ** 2 < random.random():
                        action = Action.SKIP

            data['__index'][package] = {
                'pathcomponent': pathcomponent,
                'action': action,
                'reason': reason,
            }

            if action.actionable():
                pkg_uries.append(
                    {
                        "uri": (self.repo_uri + "pypi/" + pathcomponent
                                + "/json"),
                        "parser": self.parse_package_page,
                        "output": package,
                        "timeout": 2
                    }
                )

        pkg_uries = self.decode_download_uries(pkg_uries)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=2*os.cpu_count()) as executor:
            _logger.info(f"Retrieving individual package info"
                         f" ({len(pkg_uries)} entries).")
            modulus = len(pkg_uries) // 200
            if modulus < 1:
                modulus = 1
            normal = self.do_download
            dotted = print_progress_dot(normal)
            futures = [
                executor.submit(normal if idx % modulus != 0 else dotted, uri)
                for idx, uri in enumerate(pkg_uries)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    data.update(future.result())
                except Exception:
                    # Simply swallow exceptions, there should be few enough
                    # and they will be rectified in the next run
                    pass
            print()  # Terminate the dots
        return data

    def parse_package_page(self, data_f):
        """
        Parse package page.
        """
        return json.load(data_f)

    @staticmethod
    def name_output(package, filtered_package):
        ret = package
        if package != filtered_package:
            ret += f" (as {filtered_package})"
        return ret

    def may_add_package(self, pkg_db, package, data):
        nout = self.name_output(data['realname'], package.name)
        if pkg_db.in_category(package.category, package.name):
            _logger.warn(f"Rejected package {nout} for collision.")
            self.stats['rejected'] += 1
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

        self.stats = collections.defaultdict(lambda: 0)

        for package, notes in data["packages"]["__index"].items():
            if notes['action'] is Action.EXCLUDE:
                _logger.info(f'Excluded package {package}.')
                self.stats['excluded'] += 1
                continue
            if notes['action'] is Action.IGNORE:
                _logger.info(f'Ignored package {package}'
                             f' -- {notes["reason"]}.')
                self.stats['ignored'] += 1
                if self.lookup_previous(package):
                    _logger.info(f'Preexisting entry for ignored'
                                 f' package {package}.')
                continue
            if notes['action'] is Action.SKIP:
                previous = self.lookup_previous(package)
                filtered_package, filtered_version = (
                    self.previous_package_version(package))
                if self.may_add_package(
                        pkg_db,
                        Package(category, filtered_package, filtered_version),
                        previous):
                    nout = self.name_output(package, filtered_package)
                    _logger.info(f'Skipped package {nout}.')
                    self.stats['skipped'] += 1
                continue
            if notes['action'] is Action.SKIMP:
                previous = self.lookup_previous(package)
                if previous:
                    filtered_package, filtered_version = (
                        self.previous_package_version(package))
                    if self.may_add_package(
                            pkg_db,
                            Package(category, filtered_package,
                                    filtered_version),
                            previous):
                        nout = self.name_output(package, filtered_package)
                        _logger.info(f'Skimped (preserved) package {nout}.')
                        self.stats['skimped-preserved'] += 1
                else:
                    _logger.info(f'Skimped (omitted) package {package}.')
                    self.stats['skimped-omitted'] += 1
                continue

            try:
                pkg_data = data["packages"][package]
            except KeyError:
                # First check whether this is a spurious failure
                if previous := self.lookup_previous(package):
                    filtered_package, filtered_version = (
                        self.previous_package_version(package))
                    if self.may_add_package(
                            pkg_db,
                            Package(category, filtered_package,
                                    filtered_version),
                            previous):
                        nout = self.name_output(package, filtered_package)
                        _logger.info(f'Resurrecting package {nout}.')
                        self.stats['resurrected'] += 1
                    continue
                # This happens if it is listed in the simple API list, but is
                # otherwise non-existent.
                _logger.warn(f'Package {package} is vacuous -- dropping.')
                self.stats['vacuous'] += 1
                continue
            self.process_datum(pkg_db, common_config, config, package,
                               notes, pkg_data)

        statstr = ", ".join(f"{key}: {value}"
                            for key, value in sorted(self.stats.items()))
        _logger.info(f'Package ingestion finished (statistics: {statstr})')

    @containment
    def process_datum(self, pkg_db, common_config, config, package,
                      notes, pkg_data):
        """
        Process one parsed package datum.
        """
        category = "dev-python"

        src_uri = ""
        for download_info in pkg_data['urls']:
            if download_info['packagetype'] == 'sdist':
                src_uri = download_info['url']
                break
        if not src_uri:
            _logger.warn(f'No source distfile for {package} -- dropping.')
            self.stats['nosource'] += 1
            return

        release_dates = [
            datetime.datetime.fromisoformat(release['upload_time'])
            .replace(tzinfo=datetime.timezone.utc)
            for release_list in pkg_data['releases'].values()
            for release in release_list]
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        mtime = max(release_dates, default=epoch)

        version = pkg_data['info']['version']
        homepage = pkg_data['info']['home_page'] or ""
        if not homepage:
            purls = pkg_data['info'].get('project_urls') or {}
            for key in ["Homepage", "homepage"]:
                homepage = purls.get(key, "")
                if homepage:
                    break
        homepage = self.escape_bash_string(self.strip_characters(homepage))

        pkg_license = pkg_data['info']['license'] or ''
        pkg_license = self.strip_characters(
            (pkg_license.splitlines() or [''])[0])
        pkg_license = self.convert([common_config, config], "licenses",
                                   pkg_license)
        pkg_license = self.escape_bash_string(pkg_license)

        requires_python = extract_requires_python(
            pkg_data['info']['requires_python'])
        if not requires_python:
            _logger.warn(f'No valid python versions for {package}'
                         f' -- dropping.')
            self.stats['nopython'] += 1
            return
        py_versions = list(map(
            lambda version: f'{version.components[0]}_{version.components[1]}',
            requires_python))

        if len(py_versions) == 1:
            python_compat = '( python' + py_versions[0] + ' )'
        else:
            python_compat = '( python{' + (','.join(py_versions)) + '} )'

        requires_dist = extract_requires_dist(
            pkg_data['info']['requires_dist'], self.substitutions)

        dependencies = []
        useflags = set()
        for dep in requires_dist:
            for extra in (dep['extras'] or [""]):
                # ignore version bound (found in dep["versionbound"]) as we
                # only provide the most recent version anyway so there is no
                # choice. Additionally this fixes broken dependency specs
                # where there either is an error or which are simply outdated.
                dependencies.append(Dependency(
                    category, dep['name'], usedep='${PYTHON_USEDEP}',
                    useflag=extra))
                if extra:
                    useflags.add(extra)

        filtered_package = filter_package_name(package, self.substitutions)

        filtered_description = self.escape_bash_string(self.strip_characters(
            pkg_data['info']['summary'] or ''))
        filtered_long_description = self.strip_characters(
            pkg_data['info']['description'] or '')
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
                         f' use {filtered_version}.')
        nice_src_uri = src_uri
        src_uri_filters = [(f'{package}', '${REALNAME}'),
                           (f'{version}', '${REALVERSION}')]
        for pattern, replacement in src_uri_filters:
            nice_src_uri = nice_src_uri.replace(pattern, replacement)
        filename = nice_src_uri.split('/')[-1]
        if (filename.startswith('${REALNAME}-${REALVERSION}')
                and nice_src_uri.startswith('https://files.pythonhosted.org'
                                            '/packages/')
                and package[0] in string.ascii_letters + string.digits):
            # Use redirect URL to avoid churn through the embedded hashes in
            # the actual URL
            nice_src_uri = (f'https://files.pythonhosted.org/packages/source'
                            f'/{package[0]}/${{REALNAME}}/{filename}')

        ebuild_data = {}
        ebuild_data["realname"] = package
        ebuild_data["realversion"] = version
        ebuild_data["mtime"] = mtime.isoformat()

        ebuild_data["description"] = filtered_description
        ebuild_data["longdescription"] = filtered_long_description

        ebuild_data["homepage"] = homepage
        ebuild_data["license"] = pkg_license
        ebuild_data["src_uri"] = nice_src_uri
        ebuild_data["sourcefile"] = nice_src_uri.split('/')[-1]
        ebuild_data["repo_uri"] = nice_src_uri.removesuffix(
            ebuild_data["sourcefile"])
        ebuild_data["python_compat"] = python_compat
        ebuild_data["iuse"] = " ".join(sorted(useflags))
        deplist = serializable_elist(separator="\n\t")
        deplist.extend(dependencies)
        ebuild_data["dependencies"] = deplist

        if self.may_add_package(
                pkg_db,
                Package(category, filtered_package, filtered_version),
                ebuild_data):
            nout = self.name_output(package, filtered_package)
            if notes['action'] is Action.ADD:
                _logger.info(f"Added package {nout}.")
                self.stats['added'] += 1
            elif notes['action'] is Action.UPDATE:
                _logger.info(f"Updated package {nout}.")
                self.stats['updated'] += 1
            else:
                _logger.warn(f"Processed package {nout}.")
                self.stats['unknown'] += 1

    def convert_internal_dependency(self, configs, dependency):
        """
        At the moment we have only internal dependencies, each of them
        is just a package name.
        """
        return Dependency("dev-python", dependency)
