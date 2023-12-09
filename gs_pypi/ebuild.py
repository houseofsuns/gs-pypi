#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    ebuild.py
    ~~~~~~~~~

    ebuild generation

    :copyright: (c) 2013 by Jauhien Piatlicki
    :license: GPL-2, see LICENSE for more details.
"""

import collections

from g_sorcery.ebuild import DefaultEbuildGenerator

Layout = collections.namedtuple(
    "Layout", ["vars_before_inherit", "inherit", "vars_after_description",
               "vars_after_keywords"])


class PypiEbuildWithoutDigestGenerator(DefaultEbuildGenerator):
    """
    Implementation of ebuild generator without sources digesting.
    """
    def __init__(self, package_db):
        vars_before_inherit = [
            "realname", "literalname", "realversion", "repo_uri", "sourcefile",
            {"name": "python_compat", "raw": True},
            {
                "name": "distutils_use_pep517",
                "value": "standalone",
                "raw": True
            },
        ]

        inherit = ["python-r1", "gs-pypi"]

        vars_after_description = [
            "homepage", "license",
            {
                "name": "restrict",
                "value": "test",
            }]

        vars_after_keywords = [
            "iuse", "dependencies",
            {"name": "bdepend", "value": "${DEPENDENCIES}"},
            {"name": "rdepend", "value": "${DEPENDENCIES}"}]

        layout = Layout(vars_before_inherit, inherit, vars_after_description,
                        vars_after_keywords)

        super(PypiEbuildWithoutDigestGenerator, self).__init__(package_db,
                                                               layout)


class PypiEbuildWithDigestGenerator(DefaultEbuildGenerator):
    """
    Implementation of ebuild generator with sources digesting.
    """
    def __init__(self, package_db):
        vars_before_inherit = [
            "realname", "literalname", "realversion",
            {"name": "digest_sources", "value": "yes"},
            {"name": "python_compat", "raw": True},
            {
                "name": "distutils_use_pep517",
                "value": "standalone",
                "raw": True
            },
        ]

        inherit = ["python-r1", "gs-pypi"]

        vars_after_description = [
            "homepage", "license", "src_uri", "sourcefile",
            {
                "name": "restrict",
                "value": "test",
            }]

        vars_after_keywords = [
            "iuse", "dependencies",
            {"name": "bdepend", "value": "${DEPENDENCIES}"},
            {"name": "rdepend", "value": "${DEPENDENCIES}"}]

        layout = Layout(vars_before_inherit, inherit, vars_after_description,
                        vars_after_keywords)

        super(PypiEbuildWithDigestGenerator, self).__init__(package_db, layout)
