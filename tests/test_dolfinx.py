"""The DOLFINx build: the feature set the wheel promises, and abi3 bindings."""

from pathlib import Path

import pytest

from wheelbuild import bindings, dolfinx, mpich, petsc

SONAMES = {
    "MPI": mpich.MPI_SONAME,
    "PETSc": petsc.soname(),
    "KaHIP": "libkahip.so",
}

GOOD_BUILD = dolfinx.Build(
    version=dolfinx.RELEASE,
    defines=frozenset(dolfinx.REQUIRED_DEFINES) | {"DOLFINX_VERSION"},
    runpath=(
        *dolfinx.LIBRARY_RPATH_ENTRIES,
        "/build/venv/lib/python3.12/site-packages/basix/lib",
    ),
    needed=frozenset(SONAMES.values()) | {"libc.so.6"},
)

GOOD_STAGED = dolfinx.Staged(
    runpath=dolfinx.RELATIVE_RPATH_ENTRIES,
    needed=frozenset({dolfinx.soname(), "libc.so.6"}),
)

# The domain strings nanobind compiles into an extension: what fenics-basix
# 0.11.0 carries, and what nanobind 3 writes instead.
BASIX_ABI_TAG = "v19_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1_stable"
NANOBIND_3_ABI_TAG = (
    "nanobind_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1_a1_v22_stable"
)

PKGCONFIG = """
prefix=/build/install
definitions= -DHAS_ADIOS2 -DHAS_PETSC -DHAS_SLEPC -DHAS_PTSCOTCH -DHAS_KAHIP \
-DHAS_SUPERLU_DIST -DDOLFINX_VERSION="0.11.0"

Name: DOLFINx
Version: 0.11.0
Cflags: -O2 ${definitions} -I${includedir}
"""


def _staged(site: Path) -> Path:
    """Lay out a staged DOLFINx the way an unpacked wheel leaves it."""
    for relative in dolfinx.required_staged_artefacts():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return site


def test_the_release_built_is_the_release_the_wheel_publishes():
    """Spec §3: the version mirrors the upstream release it ships."""
    from wheelbuild import version

    assert dolfinx.DOLFINX_VERSION == version.DOLFINX_VERSION
    assert dolfinx.source_url().endswith(f"v{version.DOLFINX_VERSION}.tar.gz")
    assert dolfinx.source_dir_name() == f"dolfinx-{version.DOLFINX_VERSION}"


def test_the_core_library_is_versioned_by_series():
    """`libdolfinx.so.0.11` is what the bindings ask the loader for."""
    assert dolfinx.soname("0.11.0") == "libdolfinx.so.0.11"


def test_the_extension_is_the_limited_api_one():
    """`cpp.abi3.so`, not `cpp.cpython-312-x86_64-linux-gnu.so`."""
    assert Path("dolfinx/cpp.abi3.so") == dolfinx.EXTENSION_PATH


def test_the_relative_rpath_climbs_one_level_not_two():
    """The bindings sit in the package root; petsc4py's sit in a lib/ below."""
    depth = len(dolfinx.EXTENSION_PATH.parent.parts)
    climb = "$ORIGIN/" + "../" * depth + "dolfinx_solver/lib"

    assert climb == dolfinx.RELATIVE_RPATH_ENTRIES[0]
    assert bindings.RELATIVE_RPATH not in dolfinx.RELATIVE_RPATH_ENTRIES


def test_the_extension_reaches_basix_where_its_own_wheel_installs_it():
    """The trio is depended on, not re-vendored: this points at basix's copy."""
    assert "$ORIGIN/../basix/lib" in dolfinx.RELATIVE_RPATH_ENTRIES
    assert "$ORIGIN/../../basix/lib" in dolfinx.LIBRARY_RPATH_ENTRIES


def test_configure_names_every_optional_package():
    """DOLFINx defaults them all ON and downgrades a missing one to a log."""
    arguments = dolfinx.cpp_configure_arguments(
        source_dir=Path("/build/dolfinx/cpp"),
        build_dir=Path("/build/dolfinx-cpp-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
        python="/build/venv/bin/python",
    )

    for package in ("PETSC", "SLEPC", "ADIOS2", "SCOTCH", "KAHIP", "SUPERLU_DIST"):
        assert f"-DDOLFINX_ENABLE_{package}=ON" in arguments
    assert "-DDOLFINX_ENABLE_PARMETIS=OFF" in arguments


def test_configure_takes_basix_and_ufcx_from_the_build_venv():
    """The upstream trio is depended on, never re-vendored (spec §10)."""
    arguments = dolfinx.cpp_configure_arguments(
        source_dir=Path("/build/dolfinx/cpp"),
        build_dir=Path("/build/dolfinx-cpp-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
        python="/build/venv/bin/python",
    )

    assert "-DPython3_EXECUTABLE=/build/venv/bin/python" in arguments
    assert "-DDOLFINX_BASIX_PYTHON=ON" in arguments
    assert "-DDOLFINX_UFCX_PYTHON=ON" in arguments


def test_the_environment_points_pkg_config_at_the_prefixs_petsc_and_slepc():
    """DOLFINx finds both through pkg-config, from the environment."""
    environment = dolfinx.cpp_build_environment(
        prefix=Path("/build/install"), environ={}
    )

    assert environment["PETSC_DIR"] == "/build/install"
    assert environment["SLEPC_DIR"] == "/build/install"
    assert environment["PKG_CONFIG_PATH"] == "/build/install/lib/pkgconfig"


def test_an_inherited_petsc_arch_is_cleared_rather_than_left_alone():
    environment = dolfinx.cpp_build_environment(
        prefix=Path("/build/install"), environ={"PETSC_ARCH": "arch-somebody-elses"}
    )

    assert environment["PETSC_ARCH"] == ""


def test_the_import_check_of_this_stage_asks_for_dolfinx_too():
    """The bindings stage runs the same check without it, before it exists."""
    arguments = dolfinx.import_check_arguments(
        site=Path("/build/install/python"), python="/build/venv/bin/python"
    )

    assert arguments[-1] == "--dolfinx"
    assert "wheelbuild.import_check" in arguments


def test_the_wheel_build_asks_for_a_cp312_limited_api_extension():
    """The one setting that turns on STABLE_ABI and the limited-API casters."""
    arguments = dolfinx.wheel_arguments(
        source_dir=Path("/build/dolfinx/python"),
        build_dir=Path("/build/dolfinx-python-build"),
        prefix=Path("/build/install"),
        wheelhouse=Path("/build/install/wheelhouse"),
        python="/build/venv/bin/python",
    )

    assert f"wheel.py-api={bindings.LIMITED_API_TAG}" in arguments
    assert "--no-build-isolation" in arguments
    assert "--no-deps" in arguments


def test_the_library_rpath_keeps_what_cmake_found_and_leads_with_origin():
    """The absolute path is what resolves during the build, and is kept."""
    absolute = "/build/venv/lib/python3.12/site-packages/basix/lib"

    rpath = dolfinx.library_rpath((absolute,))

    assert rpath.split(":") == [*dolfinx.LIBRARY_RPATH_ENTRIES, absolute]


def test_the_library_rpath_does_not_repeat_an_entry_already_there():
    rpath = dolfinx.library_rpath(("$ORIGIN", "/somewhere"))

    assert rpath.split(":") == [*dolfinx.LIBRARY_RPATH_ENTRIES, "/somewhere"]


def test_the_features_are_read_out_of_the_installed_pkgconfig_file():
    assert dolfinx.pkgconfig_field(PKGCONFIG, "Version") == "0.11.0"
    assert dolfinx.pkgconfig_defines(PKGCONFIG) >= frozenset(dolfinx.REQUIRED_DEFINES)


def test_a_definition_with_a_value_is_recorded_by_name():
    """``-DDOLFINX_VERSION="0.11.0"`` is the macro DOLFINX_VERSION."""
    assert "DOLFINX_VERSION" in dolfinx.pkgconfig_defines(PKGCONFIG)


def test_a_pkgconfig_file_without_the_field_reports_none():
    assert dolfinx.pkgconfig_field("Name: DOLFINx\n", "Version") is None
    assert dolfinx.pkgconfig_defines("Name: DOLFINx\n") == frozenset()


def test_a_build_that_matches_on_every_count_passes():
    assert dolfinx.build_problem(GOOD_BUILD, sonames=SONAMES) is None


def test_a_dolfinx_of_another_release_than_the_wheel_ships_is_reported():
    problem = dolfinx.build_problem(
        GOOD_BUILD._replace(version="0.10.0"), sonames=SONAMES
    )

    assert problem is not None
    assert "0.10.0" in problem


def test_a_feature_that_was_enabled_but_not_found_is_reported():
    """DOLFINx's own answer to a missing package is a log line, not an error."""
    problem = dolfinx.build_problem(
        GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {"HAS_ADIOS2"}),
        sonames=SONAMES,
    )

    assert problem is not None
    assert "ADIOS2" in problem


def test_a_dolfinx_that_found_parmetis_is_reported():
    problem = dolfinx.build_problem(
        GOOD_BUILD._replace(defines=GOOD_BUILD.defines | {"HAS_PARMETIS"}),
        sonames=SONAMES,
    )

    assert problem is not None
    assert "ParMETIS" in problem


def test_a_libdolfinx_not_bound_to_the_vendored_petsc_is_reported():
    problem = dolfinx.build_problem(
        GOOD_BUILD._replace(needed=frozenset({mpich.MPI_SONAME, "libkahip.so"})),
        sonames=SONAMES,
    )

    assert problem is not None
    assert petsc.soname() in problem


def test_the_staged_extension_that_matches_passes():
    assert dolfinx.staged_problem(GOOD_STAGED) is None


def test_a_staged_extension_with_the_build_path_on_its_rpath_is_reported():
    problem = dolfinx.staged_problem(
        GOOD_STAGED._replace(
            runpath=(*dolfinx.RELATIVE_RPATH_ENTRIES, "/build/install/lib")
        )
    )

    assert problem is not None
    assert "/build/install/lib" in problem


def test_a_staged_extension_without_the_relative_rpath_is_reported():
    problem = dolfinx.staged_problem(GOOD_STAGED._replace(runpath=()))

    assert problem is not None
    assert dolfinx.RELATIVE_RPATH in problem


def test_a_staged_extension_not_bound_to_the_core_is_reported():
    problem = dolfinx.staged_problem(
        GOOD_STAGED._replace(needed=frozenset({"libc.so.6"}))
    )

    assert problem is not None
    assert dolfinx.soname() in problem


def test_the_staging_tree_keeps_the_distribution_metadata():
    """dolfinx/__init__.py reads its own version out of it, and raises without."""
    package, dist_info = dolfinx.staged_paths()

    assert package == Path("dolfinx")
    assert dist_info.name.startswith("fenics_dolfinx-")
    assert dist_info / "METADATA" in dolfinx.required_staged_artefacts()


def test_validate_staged_returns_the_site_when_everything_holds(tmp_path, monkeypatch):
    site = _staged(tmp_path)
    monkeypatch.setattr(dolfinx, "observe_staged", lambda _: GOOD_STAGED)
    monkeypatch.setattr(dolfinx, "nanobind_abi_tag", lambda _: BASIX_ABI_TAG)

    assert dolfinx.validate_staged(site) is site


def test_validate_staged_refuses_bindings_basix_cannot_talk_to(tmp_path, monkeypatch):
    """The staged tree is complete and the rpath is right; the ABIs are not."""
    site = _staged(tmp_path)
    monkeypatch.setattr(dolfinx, "observe_staged", lambda _: GOOD_STAGED)
    tags = {dolfinx.EXTENSION_PATH.name: NANOBIND_3_ABI_TAG}
    monkeypatch.setattr(
        dolfinx,
        "nanobind_abi_tag",
        lambda path: tags.get(Path(path).name, BASIX_ABI_TAG),
    )

    with pytest.raises(ValueError, match="nanobind"):
        dolfinx.validate_staged(site)


def test_validate_staged_refuses_a_tree_without_the_vendored_library(tmp_path):
    site = _staged(tmp_path)
    (site / bindings.VENDORED_LIBRARY_DIR / dolfinx.soname()).unlink()

    with pytest.raises(FileNotFoundError, match="libdolfinx"):
        dolfinx.validate_staged(site)


def test_validate_refuses_an_incomplete_cpp_install(tmp_path):
    with pytest.raises(FileNotFoundError, match="libdolfinx"):
        dolfinx.validate_cpp(tmp_path)


def test_the_upstream_trio_is_read_back_out_of_our_own_metadata():
    """One place declares the pins, and the build installs those (spec §10)."""
    requirements = dolfinx.upstream_trio_requirements()

    assert len(requirements) == len(dolfinx.UPSTREAM_TRIO)
    for name, requirement in zip(dolfinx.UPSTREAM_TRIO, requirements, strict=True):
        assert requirement.startswith(name)


def test_a_project_that_does_not_declare_the_trio_is_refused():
    with pytest.raises(LookupError, match="fenics-basix"):
        dolfinx.upstream_trio_requirements('[project]\ndependencies = ["numpy"]\n')


def test_the_staging_site_stands_in_for_an_installed_basix(tmp_path, monkeypatch):
    """`$ORIGIN/../basix/lib` has to resolve for the import check to mean anything."""
    basix = tmp_path / "venv" / "basix" / "lib"
    basix.mkdir(parents=True)
    (basix / "libbasix.so").touch()
    monkeypatch.setattr(dolfinx, "basix_library_dir", lambda _python: basix)

    site = dolfinx.prepare_site(tmp_path / "install")

    assert (site / dolfinx.BASIX_LIBRARY_DIR / "libbasix.so").exists()
    assert (site / dolfinx.BASIX_LIBRARY_DIR).is_symlink()


def test_a_real_basix_directory_in_the_payload_is_refused(tmp_path, monkeypatch):
    """A copy of basix inside the wheel is not what this build produces."""
    monkeypatch.setattr(dolfinx, "basix_library_dir", lambda _python: tmp_path)
    site = bindings.site_dir(tmp_path / "install")
    (site / dolfinx.BASIX_LIBRARY_DIR).mkdir(parents=True)

    with pytest.raises(NotADirectoryError, match="fenics-basix"):
        dolfinx.prepare_site(tmp_path / "install")


def _extension(path: Path, tag: str) -> Path:
    """Write a stand-in extension carrying one nanobind ABI tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF\x00" + tag.encode() + b"\x00padding")
    return path


def test_the_nanobind_abi_tag_is_read_out_of_the_binary(tmp_path):
    """What a wheel on PyPI was built with is knowable only from the wheel."""
    extension = _extension(tmp_path / "_basixcpp.abi3.so", BASIX_ABI_TAG)

    assert dolfinx.nanobind_abi_tag(extension) == BASIX_ABI_TAG


def test_an_extension_without_a_tag_reports_none(tmp_path):
    extension = tmp_path / "plain.so"
    extension.write_bytes(b"\x7fELF\x00no nanobind here\x00")

    assert dolfinx.nanobind_abi_tag(extension) is None


def test_matching_nanobind_abis_are_what_lets_basix_types_cross():
    assert dolfinx.nanobind_problem(BASIX_ABI_TAG, BASIX_ABI_TAG) is None


def test_a_newer_nanobind_than_the_basix_wheels_is_reported():
    """Otherwise it surfaces as a TypeError in the first functionspace call."""
    problem = dolfinx.nanobind_problem(NANOBIND_3_ABI_TAG, BASIX_ABI_TAG)

    assert problem is not None
    assert dolfinx.NANOBIND_REQUIREMENT in problem


def test_an_untagged_extension_is_reported_rather_than_passed():
    assert dolfinx.nanobind_problem(None, None) is not None


def test_a_core_library_without_its_relative_rpath_is_reported():
    """The absolute paths CMake baked in resolve here and nowhere else."""
    problem = dolfinx.build_problem(
        GOOD_BUILD._replace(
            runpath=("/build/venv/lib/python3.12/site-packages/basix/lib",)
        ),
        sonames=SONAMES,
    )

    assert problem is not None
    assert "$ORIGIN" in problem


def test_libdolfinx_is_checked_against_superlu_dist_too():
    """Every required feature has a library the linkage check can name."""
    assert set(dolfinx.REQUIRED_DEFINES.values()) <= set(dolfinx.REQUIRED_LINKAGE)


#: A resolution of the pins this stage compiles against, as the build venv
#: would report them: exact releases, not the bounds the drivers declare.
RESOLVED = {
    "nanobind": "2.12.1",
    "fenics-basix": "0.11.0",
    "fenics-ffcx": "0.11.0",
    "fenics-ufl": "2025.2.0",
}


def test_the_derived_pins_this_stage_compiles_against_are_nanobind_and_the_trio():
    """The mpich bound is the third derived pin and is not one of these: what
    this stage links is the MPICH in the prefix (spec §5, ADR-0001)."""
    assert ("nanobind", *dolfinx.UPSTREAM_TRIO) == dolfinx.COMPILED_DERIVED_PINS


def test_the_derived_pin_id_is_a_cache_key_the_stamp_can_carry():
    """It goes in a file name, so it is short hex like the build
    script's `tooling_id` rather than a whole digest."""
    identifier = dolfinx.derived_pin_id(RESOLVED)

    assert len(identifier) == dolfinx.DERIVED_PIN_ID_LENGTH
    assert set(identifier) <= set("0123456789abcdef")


def test_the_same_resolution_is_the_same_identifier():
    """A stamp that moved without an input moving would rebuild every run."""
    assert dolfinx.derived_pin_id(RESOLVED) == dolfinx.derived_pin_id(dict(RESOLVED))


def test_a_patch_release_inside_the_declared_bound_moves_the_identifier():
    """`nanobind==2.12.*` resolves to 2.12.1 today and 2.12.2 tomorrow, and
    the bindings are compiled by whichever it is (ticket 31)."""
    bumped = RESOLVED | {"nanobind": "2.12.2"}

    assert dolfinx.derived_pin_id(bumped) != dolfinx.derived_pin_id(RESOLVED)


def test_a_fresh_basix_inside_the_declared_bound_moves_the_identifier():
    """The C++ core compiles against Basix's headers, so a warm prefix holds
    one built against the previous release until the stamp notices."""
    bumped = RESOLVED | {"fenics-basix": "0.11.1"}

    assert dolfinx.derived_pin_id(bumped) != dolfinx.derived_pin_id(RESOLVED)


def test_the_identifier_does_not_depend_on_the_order_they_were_read_in():
    """The key is about which releases are installed; a stamp that moved
    because a caller iterated differently would recompile for nothing."""
    reversed_order = dict(reversed(list(RESOLVED.items())))

    assert dolfinx.derived_pin_id(reversed_order) == dolfinx.derived_pin_id(RESOLVED)


def test_the_resolved_pins_are_read_out_of_the_environment_they_installed_into():
    """The build script runs this through the build venv's interpreter, which
    is where the three `pip install` lines put them."""
    resolved = dolfinx.resolved_derived_pins(("pytest",))

    assert set(resolved) == {"pytest"}
    assert resolved["pytest"]


def test_a_pin_that_is_not_installed_is_refused_rather_than_hashed():
    """Hashing an absent release would key the stamp on a lie: the stage
    would build against whatever pip resolved and claim it did not."""
    with pytest.raises(LookupError, match="not-installed-here"):
        dolfinx.resolved_derived_pins(("not-installed-here",))
