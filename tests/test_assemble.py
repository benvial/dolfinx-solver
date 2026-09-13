"""The wheel assembly: what goes in, what is refused, and what is checked.

The functions that touch binaries — the closure walk, the pruning, the
validation — are checked here against recorded facts and made-up trees rather
than against a built prefix, the way the rest of the drivers are tested. What
the container run proves is that those facts are the real ones.
"""

import base64
import hashlib
import zipfile
from pathlib import Path

import pytest

from wheelbuild import assemble, bindings, dolfinx, mpich, notices, petsc, version

LIBRARY_DIR = bindings.VENDORED_LIBRARY_DIR.as_posix()

#: The shipped members a publishable wheel has, as the validation wants them.
GOOD_NAMES = (
    f"{assemble.dist_info_name()}/METADATA",
    f"{assemble.dist_info_name()}/WHEEL",
    f"{assemble.dist_info_name()}/RECORD",
    "dolfinx/__init__.py",
    "dolfinx/cpp.abi3.so",
    "dolfinx_solver/__init__.py",
    assemble.NOTICES_PATH.as_posix(),
    "petsc4py/lib/PETSc.abi3.so",
    "slepc4py/lib/SLEPc.abi3.so",
    f"{LIBRARY_DIR}/{dolfinx.soname()}",
    f"{LIBRARY_DIR}/{petsc.soname()}",
    f"{LIBRARY_DIR}/{mpich.FORTRAN_SONAME}",
    f"{assemble.GRAFT_DIR}/libgfortran-a1b2c3d4.so.5",
)


def _good_wheel(**overrides) -> assemble.Wheel:
    """Return what a publishable wheel says about itself."""
    facts = {
        "names": GOOD_NAMES,
        "tag": assemble.WHEEL_TAG,
        "libraries": assemble.shipped_libraries(GOOD_NAMES),
        "runpaths": {
            "dolfinx/cpp.abi3.so": dolfinx.RELATIVE_RPATH_ENTRIES,
            f"{LIBRARY_DIR}/{dolfinx.soname()}": dolfinx.LIBRARY_RPATH_ENTRIES,
            "petsc4py/lib/PETSc.abi3.so": (bindings.RELATIVE_RPATH,),
            "slepc4py/lib/SLEPc.abi3.so": (bindings.RELATIVE_RPATH,),
        },
        "needed": {
            f"{LIBRARY_DIR}/{petsc.soname()}": frozenset(
                {"libmpi.so.12", mpich.FORTRAN_SONAME, "libc.so.6"}
            ),
            f"{LIBRARY_DIR}/{mpich.FORTRAN_SONAME}": frozenset(
                {"libmpi.so.12", "libgfortran.so.5", "libc.so.6"}
            ),
        },
        "build_paths": (),
    }
    facts.update(overrides)
    return assemble.Wheel(**facts)


def test_the_wheel_is_named_and_tagged_the_way_the_spec_says():
    """One abi3 wheel per release, manylinux_2_34 (spec §2, §4)."""
    assert assemble.WHEEL_TAG == "cp312-abi3-manylinux_2_34_x86_64"
    assert assemble.wheel_name(assemble.WHEEL_TAG) == (
        f"dolfinx_solver_{petsc.SCALAR_TYPE}-{version.__version__}"
        "-cp312-abi3-manylinux_2_34_x86_64.whl"
    )


def test_the_assembled_wheel_is_tagged_before_auditwheel_has_ruled():
    """A freshly-linked binary is ``linux_x86_64`` until repair says otherwise."""
    assert assemble.BUILD_TAG.endswith("-linux_x86_64")
    assert assemble.BUILD_TAG.startswith("cp312-abi3")


def test_the_wheel_metadata_is_not_purelib():
    """A purelib wheel would install the libraries away from the rpaths."""
    metadata = assemble.wheel_metadata(assemble.BUILD_TAG)

    assert "Root-Is-Purelib: false" in metadata
    assert f"Tag: {assemble.BUILD_TAG}" in metadata


def test_the_two_libraries_another_wheel_owns_are_excluded():
    """libmpi from the mpich wheel, libbasix from fenics-basix (spec §5, §10)."""
    assert assemble.excluded("libmpi.so.12")
    assert assemble.excluded("libbasix.so")
    assert not assemble.excluded("libpetsc.so.3.25")
    assert not assemble.excluded(mpich.FORTRAN_SONAME)


def test_the_repair_names_every_exclusion_on_the_command_line():
    arguments = assemble.repair_arguments(Path("/build/w.whl"), Path("/wheelhouse"))

    assert arguments[:2] == ["auditwheel", "repair"]
    for pattern in assemble.EXCLUDED_SONAMES:
        assert "--exclude" in arguments
        assert pattern in arguments
    assert assemble.PLATFORM_TAG in arguments


def test_the_platform_report_is_asked_for_as_json():
    """Prose cannot be checked by a build."""
    assert "--json" in assemble.show_arguments(Path("/wheelhouse/w.whl"))


def test_a_wheel_whose_symbols_fit_the_platform_it_claims_is_accepted():
    """The two unresolved externals are the ones other wheels own (spec §5, §10)."""
    assert (
        assemble.platform_problem(
            {
                "sym_tag": assemble.PLATFORM_TAG,
                "external_libs": {"libmpi.so.12": None, "libbasix.so": None},
                "policy_upgrades": {
                    assemble.PLATFORM_TAG: {
                        "libs_to_eliminate": ["libbasix.so", "libmpi.so.12"]
                    }
                },
            }
        )
        is None
    )


def test_a_library_the_claimed_platform_allows_is_not_read_as_a_fault():
    """``external_libs`` is computed against a stricter policy than ours."""
    assert (
        assemble.platform_problem(
            {
                "sym_tag": assemble.PLATFORM_TAG,
                "external_libs": {"libexpat.so.1": "/usr/lib64/libexpat.so.1"},
                "policy_upgrades": {
                    "manylinux_2_5_x86_64": {"libs_to_eliminate": ["libexpat.so.1"]}
                },
            }
        )
        is None
    )


def test_a_wheel_built_against_the_wrong_glibc_is_refused():
    problem = assemble.platform_problem(
        {"sym_tag": "manylinux_2_39_x86_64", "external_libs": {}}
    )

    assert problem is not None
    assert "manylinux_2_39_x86_64" in problem


def test_an_external_dependency_nothing_else_provides_is_refused():
    """A library found on the builder and nowhere else is a failed import."""
    problem = assemble.platform_problem(
        {
            "sym_tag": assemble.PLATFORM_TAG,
            "policy_upgrades": {
                assemble.PLATFORM_TAG: {
                    "libs_to_eliminate": ["libmpi.so.12", "libucp.so.0"]
                }
            },
        }
    )

    assert problem is not None
    assert "libucp.so.0" in problem


def test_the_abi3_gate_is_strict():
    """Spec §4: `abi3audit --strict` on every built wheel."""
    assert "--strict" in assemble.abi3audit_arguments(Path("/wheelhouse/w.whl"))


def test_pruning_drops_the_transport_line_from_the_mpich_halves():
    """MPICH's wrappers link UCX into libraries that never call it."""
    needed = [
        "libmpi.so.12",
        "libucp.so.0",
        "libuct.so.0",
        "libucs.so.0",
        "libucm.so.0",
        "libpciaccess.so.0",
        "libatomic.so.1",
        "libgfortran.so.5",
        "libc.so.6",
    ]

    assert assemble.pruned(mpich.FORTRAN_SONAME, needed) == frozenset(
        {"libmpi.so.12", "libgfortran.so.5", "libc.so.6"}
    )
    assert assemble.pruned(mpich.CXX_SONAME, needed) == frozenset(
        {"libmpi.so.12", "libgfortran.so.5", "libc.so.6"}
    )


def test_pruning_leaves_every_other_library_alone():
    """Only the two MPICH shims are over-linked; nothing else is touched."""
    needed = ["libmpi.so.12", "libucp.so.0", "libc.so.6"]

    assert assemble.pruned(petsc.soname(), needed) == frozenset(needed)


def test_the_pruned_dependencies_are_the_ones_the_wheel_would_have_to_vendor():
    """UCX is not in the licensing table (spec §7); that is the point."""
    for soname in assemble.OVER_LINKED[mpich.FORTRAN_SONAME]:
        assert notices.unclaimed_libraries([soname]) == [soname]


def test_the_patchelf_command_removes_each_entry_by_name():
    arguments = assemble.patchelf_arguments(
        Path("/build/lib/libmpifort.so.12"), ["libucp.so.0", "libucs.so.0"]
    )

    assert arguments[0] == "patchelf"
    assert arguments.count("--remove-needed") == 2
    assert arguments[-1] == "/build/lib/libmpifort.so.12"


def test_a_dependency_that_cannot_be_found_is_not_pruned_on_faith(
    tmp_path, monkeypatch
):
    """Checking the claim is the whole licence to act on it."""
    monkeypatch.setattr(assemble, "SYSTEM_LIBRARY_DIRS", ())
    monkeypatch.setattr(assemble.elf, "read_symbols", lambda _: (set(), {"mpi_init_"}))

    problem = assemble.prune_problem(
        tmp_path / "libmpifort.so.12", ["libucp.so.0"], tmp_path
    )

    assert problem is not None
    assert "cannot be checked" in problem


def test_a_dependency_whose_symbols_are_used_is_not_pruned(tmp_path, monkeypatch):
    """A future MPICH that really calls UCX fails the build instead."""
    (tmp_path / "libucp.so.0").touch()
    symbols = {
        "libmpifort.so.12": (set(), {"ucp_worker_progress"}),
        "libucp.so.0": ({"ucp_worker_progress"}, set()),
    }
    monkeypatch.setattr(
        assemble.elf, "read_symbols", lambda path: symbols[Path(path).name]
    )

    problem = assemble.prune_problem(
        tmp_path / "libmpifort.so.12", ["libucp.so.0"], tmp_path
    )

    assert problem is not None
    assert "ucp_worker_progress" in problem


def test_an_unreferenced_dependency_is_safe_to_prune(tmp_path, monkeypatch):
    (tmp_path / "libucp.so.0").touch()
    symbols = {
        "libmpifort.so.12": (set(), {"MPI_Init", "memcpy"}),
        "libucp.so.0": ({"ucp_worker_progress"}, set()),
    }
    monkeypatch.setattr(
        assemble.elf, "read_symbols", lambda path: symbols[Path(path).name]
    )

    assert (
        assemble.prune_problem(tmp_path / "libmpifort.so.12", ["libucp.so.0"], tmp_path)
        is None
    )


def test_a_dependency_the_library_does_not_name_is_not_looked_for(
    tmp_path, monkeypatch
):
    """An MPICH whose shims are already clean must not fail the build."""
    staged = tmp_path / "lib"
    staged.mkdir()
    (staged / mpich.FORTRAN_SONAME).touch()
    (staged / mpich.CXX_SONAME).touch()
    monkeypatch.setattr(
        assemble.elf, "read_dynamic", lambda _: (None, frozenset({"libmpi.so.12"}))
    )
    monkeypatch.setattr(assemble, "prune_problem", lambda *_: "should never be asked")
    calls: list[object] = []
    monkeypatch.setattr(assemble, "check_call", calls.append)

    assert assemble.prune(tmp_path / "prefix-lib", staged) == []
    assert calls == []


def test_the_closure_follows_what_the_binaries_ask_for(tmp_path, monkeypatch):
    """The payload is the closure, not the prefix (129 libraries live there)."""
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    for name in (
        "libpetsc.so.3.25",
        "libscotch.so.7.0",
        "libmpi.so.12",
        "libunreferenced.so.1",
        mpich.FORTRAN_SONAME,
        "libucp.so.0",
    ):
        (library_dir / name).touch()
    extension = tmp_path / "PETSc.abi3.so"
    extension.touch()

    needed = {
        "PETSc.abi3.so": frozenset({"libpetsc.so.3.25", "libc.so.6"}),
        "libpetsc.so.3.25": frozenset(
            {"libscotch.so.7.0", mpich.FORTRAN_SONAME, "libmpi.so.12"}
        ),
        "libscotch.so.7.0": frozenset({"libc.so.6"}),
        mpich.FORTRAN_SONAME: frozenset({"libmpi.so.12", "libucp.so.0"}),
    }
    monkeypatch.setattr(
        assemble.elf, "read_dynamic", lambda path: (None, needed[Path(path).name])
    )

    found = assemble.closure([extension], library_dir)

    assert set(found) == {"libpetsc.so.3.25", "libscotch.so.7.0", mpich.FORTRAN_SONAME}
    assert "libmpi.so.12" not in found, "the excluded runtime is never carried"
    assert "libucp.so.0" not in found, "pruning happens before the walk reaches it"
    assert "libunreferenced.so.1" not in found, "nothing links it"


def test_the_closure_prunes_by_soname_not_by_the_file_it_resolves_to(
    tmp_path, monkeypatch
):
    """In the prefix, ``libmpifort.so.12`` is a symlink to ``...so.12.6.1``."""
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    real = library_dir / f"{mpich.FORTRAN_SONAME}.6.1"
    real.touch()
    (library_dir / mpich.FORTRAN_SONAME).symlink_to(real.name)
    (library_dir / "libucp.so.0").touch()
    extension = tmp_path / "PETSc.abi3.so"
    extension.touch()

    needed = {
        "PETSc.abi3.so": frozenset({mpich.FORTRAN_SONAME}),
        f"{mpich.FORTRAN_SONAME}.6.1": frozenset({"libucp.so.0"}),
    }
    monkeypatch.setattr(
        assemble.elf, "read_dynamic", lambda path: (None, needed[Path(path).name])
    )

    found = assemble.closure([extension], library_dir)

    assert mpich.FORTRAN_SONAME in found
    assert "libucp.so.0" not in found


def test_the_audits_do_not_run_with_the_build_prefix_on_the_library_path():
    """LD_LIBRARY_PATH beats DT_RUNPATH for the loader and for auditwheel.

    With the build prefix on it, every vendored library resolves to the prefix
    copy rather than to the one inside the wheel, is classified as external,
    and is grafted a second time under a hashed name.
    """
    environment = assemble.audit_environment(
        {"LD_LIBRARY_PATH": "/build/install/lib", "PATH": "/usr/bin"}
    )

    assert "LD_LIBRARY_PATH" not in environment
    assert environment["PATH"] == "/usr/bin"


def test_a_payload_package_without_its_extension_fails(tmp_path):
    for package in assemble.PAYLOAD_PACKAGES:
        (tmp_path / package).mkdir()

    with pytest.raises(FileNotFoundError, match="no extension module"):
        assemble.payload_extensions(tmp_path)


def test_the_build_prefix_is_emptied_out_of_the_bindings_configuration():
    """Inert at import, and still not something to publish (ticket 04)."""
    rewritten = assemble.blank_values(
        "PETSC_DIR  = /build/install\nPETSC_ARCH = \n", ["PETSC_DIR"]
    )

    assert "/build" not in rewritten
    assert rewritten == "PETSC_DIR  =\nPETSC_ARCH = \n"


def test_blanking_does_not_invent_configuration():
    """A key upstream does not write is not added by the rewrite."""
    assert assemble.blank_values("PETSC_ARCH = \n", ["SLEPC_DIR"]) == "PETSC_ARCH = \n"


def test_a_record_line_carries_the_hash_and_the_size(tmp_path):
    path = tmp_path / "thing.py"
    path.write_bytes(b"x = 1\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(b"x = 1\n").digest()).rstrip(b"=")

    assert assemble.record_entry(path, "thing.py") == (
        f"thing.py,sha256={digest.decode()},6"
    )


def test_the_record_lists_every_file_but_hashes_itself_empty(tmp_path):
    dist_info = assemble.dist_info_name()
    (tmp_path / dist_info).mkdir()
    (tmp_path / dist_info / "METADATA").write_text("Name: x\n")
    (tmp_path / "dolfinx_solver").mkdir()
    (tmp_path / "dolfinx_solver" / "__init__.py").write_text("")

    record = assemble.write_record(tmp_path, dist_info)
    lines = record.read_text(encoding="utf-8").splitlines()

    assert f"{dist_info}/RECORD,," in lines
    assert any(line.startswith("dolfinx_solver/__init__.py,sha256=") for line in lines)
    assert any(line.startswith(f"{dist_info}/METADATA,sha256=") for line in lines)


def test_packing_and_unpacking_keep_the_executable_bit(tmp_path):
    """A wheel's libraries are executable, and zipfile drops modes."""
    tree = tmp_path / "tree"
    (tree / "dolfinx_solver" / "lib").mkdir(parents=True)
    library = tree / "dolfinx_solver" / "lib" / "libpetsc.so.3.25"
    library.write_bytes(b"\x7fELF stand-in")
    library.chmod(0o755)
    (tree / "dolfinx_solver" / "__init__.py").write_text("")

    archive = assemble.pack(tree, tmp_path / "wheel.whl")
    restored = assemble.unpack(archive, tmp_path / "back")

    with zipfile.ZipFile(archive) as zipped:
        assert sorted(zipped.namelist()) == [
            "dolfinx_solver/__init__.py",
            "dolfinx_solver/lib/libpetsc.so.3.25",
        ]
    assert (restored / "dolfinx_solver/lib/libpetsc.so.3.25").stat().st_mode & 0o111


#: The two lines of the checked-in ``pyproject.toml`` that name the variant.
PROJECT_METADATA = (
    'name = "dolfinx-solver-complex"\n'
    'description = "DOLFINx (FEniCSx) with complex-scalar PETSc and SLEPc, '
    'packaged as a binary wheel"\n'
)


#: The passages of the checked-in ``README.md`` that name the variant. It is
#: the wheel's ``Description``, so the real wheel's project page is what these
#: decide (ticket 32).
README = (
    "DOLFINx packaged as a Linux binary wheel, with a **complex-scalar PETSc "
    "and SLEPc inside the wheel**.\n"
    "\n"
    "```console\n"
    "pip install dolfinx-solver          # the complex build, via the meta-package\n"
    "pip install dolfinx-solver-complex  # the same wheel, named directly\n"
    "```\n"
    "\n"
    'assert PETSc.ScalarType.__name__ == "complex128"\n'
    "\n"
    "DOLFINx's C++ core and nanobind bindings, complex-scalar PETSc and SLEPc "
    "with petsc4py and slepc4py built against exactly those.\n"
    "\n"
    "The real-scalar build is published separately as `dolfinx-solver-real`;\n"
    "PETSc's scalar type is baked into the binaries, so it cannot be switched\n"
    "at runtime.\n"
)


def _repository(
    root: Path, pyproject: str = PROJECT_METADATA, readme: str = README
) -> Path:
    """Lay out the files :data:`assemble.SOURCE_INPUTS` names."""
    (root / "dolfinx_solver").mkdir(parents=True, exist_ok=True)
    (root / "dolfinx_solver" / "__init__.py").write_text("")
    for name in ("LICENSE", "LICENSE.GPL-3.0"):
        (root / name).write_text("")
    (root / "README.md").write_text(readme)
    (root / "pyproject.toml").write_text(pyproject)
    return root


def test_the_source_copy_takes_the_package_without_its_build_products(tmp_path):
    """``dolfinx_solver/lib`` exists in a working tree that has built once."""
    repo = _repository(tmp_path / "repo")
    (repo / "dolfinx_solver" / "lib").mkdir(parents=True)
    (repo / "dolfinx_solver" / "lib" / "libpetsc.so.3.25").write_text("stale")
    (repo / "dolfinx_solver" / "__pycache__").mkdir()

    copied = assemble.source_copy(tmp_path / "copy", repo_root=repo)

    assert (copied / "dolfinx_solver" / "__init__.py").exists()
    assert not (copied / "dolfinx_solver" / "lib").exists()
    assert not (copied / "dolfinx_solver" / "__pycache__").exists()


def test_a_missing_source_input_fails_rather_than_building_something_else(tmp_path):
    with pytest.raises(FileNotFoundError, match="SOURCE_INPUTS"):
        assemble.source_copy(tmp_path / "copy", repo_root=tmp_path / "empty")


def test_the_libraries_are_read_out_of_wherever_the_wheel_keeps_them():
    """Ours sit in the sibling directory; auditwheel's graft has its own."""
    assert assemble.shipped_libraries(GOOD_NAMES) == (
        "libdolfinx.so.0.11",
        "libgfortran-a1b2c3d4.so.5",
        mpich.FORTRAN_SONAME,
        petsc.soname(),
    )


def test_a_publishable_wheel_has_no_problem():
    assert assemble.wheel_problem(_good_wheel()) is None


def test_a_wheel_tagged_for_one_interpreter_is_refused():
    problem = assemble.wheel_problem(_good_wheel(tag="cp312-cp312-linux_x86_64"))

    assert problem is not None
    assert "stable ABI" in problem


def test_a_grafted_libmpi_is_refused():
    """The failure the whole linkage design exists to avoid (ADR-0002)."""
    names = (*GOOD_NAMES, f"{assemble.GRAFT_DIR}/libmpi-deadbeef.so.12")
    problem = assemble.wheel_problem(
        _good_wheel(names=names, libraries=assemble.shipped_libraries(names))
    )

    assert problem is not None
    assert "second copy in the process" in problem


def test_a_grafted_libbasix_is_refused():
    names = (*GOOD_NAMES, f"{assemble.GRAFT_DIR}/libbasix-deadbeef.so")
    problem = assemble.wheel_problem(
        _good_wheel(names=names, libraries=assemble.shipped_libraries(names))
    )

    assert problem is not None
    assert "two type registries" in problem


def test_a_copied_basix_package_is_refused():
    """The staging symlink having been followed instead of skipped."""
    problem = assemble.wheel_problem(
        _good_wheel(names=(*GOOD_NAMES, "basix/__init__.py"))
    )

    assert problem is not None
    assert "fenics-basix" in problem


def test_shipped_bytecode_is_refused():
    problem = assemble.wheel_problem(
        _good_wheel(names=(*GOOD_NAMES, "dolfinx/__pycache__/__init__.cpython-312.pyc"))
    )

    assert problem is not None
    assert "abi3" in problem


def test_a_member_naming_the_container_build_prefix_is_refused():
    problem = assemble.wheel_problem(
        _good_wheel(build_paths=("petsc4py/lib/petsc.cfg",))
    )

    assert problem is not None
    assert "petsc.cfg" in problem


def test_a_missing_notice_file_is_refused():
    """Spec §7's enforcement, at the last point it can be checked."""
    names = tuple(
        name for name in GOOD_NAMES if name != assemble.NOTICES_PATH.as_posix()
    )
    problem = assemble.wheel_problem(_good_wheel(names=names))

    assert problem is not None
    assert "THIRD-PARTY-NOTICES" in problem


def test_metadata_for_a_distribution_this_project_does_not_publish_is_refused():
    """pip refuses a wheel with two .dist-info directories outright (ticket 14)."""
    names = (
        *GOOD_NAMES,
        f"fenics_dolfinx-{version.DOLFINX_VERSION}.dist-info/METADATA",
    )
    problem = assemble.wheel_problem(_good_wheel(names=names))

    assert problem is not None
    assert "fenics_dolfinx" in problem
    assert "multiple .dist-info" in problem


def test_the_packaged_dolfinx_learns_its_version_from_a_literal():
    """The metadata it read cannot ship, so the line is rewritten (ticket 14)."""
    upstream = "import sys\n\n" + assemble.VERSION_SUBSTITUTION[0] + "\n"

    substituted = assemble.substitute_version(upstream)

    assert f'__version__ = "{version.DOLFINX_VERSION}"' in substituted
    assert "importlib.metadata" not in substituted
    assert "import sys" in substituted


def test_a_dolfinx_that_reads_its_version_another_way_fails_the_build():
    """An upstream change here is an import that raises for the user."""
    with pytest.raises(ValueError, match="does not contain"):
        assemble.substitute_version("__version__ = '0.12.0'\n")


def test_the_packaged_dolfinx_finds_libpetsc_beside_the_payload():
    """Upstream reads a PETSC_DIR the wheel has no honest value for (ticket 16)."""
    upstream = (
        "def get_petsc_lib():\n"
        + assemble.PETSC_LIB_SUBSTITUTION[0]
        + "\n    return pathlib.Path(exists_paths[0])\n"
    )

    substituted = assemble.substitute_petsc_lib(upstream)

    assert f'"{LIBRARY_DIR}"' in substituted
    assert f'"{petsc.soname()}"' in substituted
    assert "get_config()" not in substituted
    assert "exists_paths" in substituted


def test_a_dolfinx_that_finds_libpetsc_another_way_fails_the_build():
    """A wheel whose solver layer cannot import is worse than no wheel."""
    with pytest.raises(ValueError, match="does not look up"):
        assemble.substitute_petsc_lib("def get_petsc_lib():\n    return None\n")


def test_two_copies_of_one_library_are_refused():
    """The failure an absolute RUNPATH in a staged library produces."""
    names = (
        *GOOD_NAMES,
        f"{assemble.GRAFT_DIR}/libpetsc-7dc9be3b.so.3.25.5",
    )
    problem = assemble.wheel_problem(
        _good_wheel(names=names, libraries=assemble.shipped_libraries(names))
    )

    assert problem is not None
    assert "two copies of libpetsc" in problem


def test_the_graft_hash_is_stripped_off_a_library_name():
    assert assemble.library_stem("libpetsc-7dc9be3b.so.3.25.5") == "libpetsc"
    assert assemble.library_stem("libpetsc.so.3.25") == "libpetsc"
    assert assemble.library_stem("libopenblas-r0-0c4a605d.3.32.so") == "libopenblas-r0"
    assert assemble.library_stem("libopenblas.so.0") == "libopenblas"


def test_a_library_versioned_before_the_so_is_still_recognised():
    """OpenBLAS is vendored as libopenblas.so.0 and built as libopenblas-r0."""
    assert assemble.same_library("libopenblas", "libopenblas-r0")
    assert assemble.same_library("libpetsc", "libpetsc")
    assert not assemble.same_library("libscotch", "libscotcherr")


def test_two_copies_of_openblas_are_refused_despite_the_two_spellings():
    names = (
        *GOOD_NAMES,
        f"{LIBRARY_DIR}/libopenblas.so.0",
        f"{assemble.GRAFT_DIR}/libopenblas-r0-0c4a605d.3.32.so",
    )
    problem = assemble.wheel_problem(
        _good_wheel(names=names, libraries=assemble.shipped_libraries(names))
    )

    assert problem is not None
    assert "two copies of libopenblas" in problem


def test_every_vendored_library_is_found_relative_to_where_it_is_installed():
    """An absolute RUNPATH is what makes auditwheel graft a second copy."""
    assert assemble.library_rpath(petsc.soname()) == "$ORIGIN"
    assert assemble.library_rpath(mpich.FORTRAN_SONAME) == "$ORIGIN"
    assert assemble.library_rpath(dolfinx.soname()) == ":".join(
        dolfinx.LIBRARY_RPATH_ENTRIES
    )


def test_a_library_no_component_accounts_for_is_refused():
    """The check that catches the library nobody chose."""
    names = (*GOOD_NAMES, f"{assemble.GRAFT_DIR}/libucp-deadbeef.so.0")
    problem = assemble.wheel_problem(
        _good_wheel(names=names, libraries=assemble.shipped_libraries(names))
    )

    assert problem is not None
    assert "libucp" in problem
    assert "licence" in problem


def test_an_rpath_the_repair_dropped_is_refused():
    """auditwheel rewrites rpaths, and the sibling layout is not its decision."""
    runpaths = dict(_good_wheel().runpaths)
    runpaths["dolfinx/cpp.abi3.so"] = (f"$ORIGIN/../{assemble.GRAFT_DIR}",)
    problem = assemble.wheel_problem(_good_wheel(runpaths=runpaths))

    assert problem is not None
    assert dolfinx.RELATIVE_RPATH_ENTRIES[0] in problem


def test_the_basix_rpath_entry_has_to_survive_the_repair():
    """Ticket 05 flagged this: the entry reaches a directory outside the wheel."""
    runpaths = dict(_good_wheel().runpaths)
    runpaths[f"{LIBRARY_DIR}/{dolfinx.soname()}"] = ("$ORIGIN",)
    problem = assemble.wheel_problem(_good_wheel(runpaths=runpaths))

    assert problem is not None
    assert dolfinx.LIBRARY_RPATH_ENTRIES[1] in problem


def test_an_absolute_rpath_entry_is_refused():
    """The build paths the earlier stages kept on purpose must not ship."""
    runpaths = dict(_good_wheel().runpaths)
    runpaths["petsc4py/lib/PETSc.abi3.so"] = (
        bindings.RELATIVE_RPATH,
        "/build/install/lib",
    )
    problem = assemble.wheel_problem(_good_wheel(runpaths=runpaths))

    assert problem is not None
    assert "/build/install/lib" in problem


def test_a_renamed_libmpi_dependency_is_refused():
    """If the exclusion is lost, libpetsc binds to a grafted copy instead."""
    needed = dict(_good_wheel().needed)
    needed[f"{LIBRARY_DIR}/{petsc.soname()}"] = frozenset(
        {"libmpi-deadbeef.so.12", "libc.so.6"}
    )
    problem = assemble.wheel_problem(_good_wheel(needed=needed))

    assert problem is not None
    assert "plain" in problem


def test_a_wheel_without_the_vendored_petsc_is_refused():
    """The MPI linkage cannot be checked on a wheel that has no libpetsc."""
    problem = assemble.wheel_problem(_good_wheel(needed={}))

    assert problem is not None
    assert "cannot be checked" in problem


def test_a_member_that_merely_names_the_build_script_is_not_a_leak(tmp_path):
    """`scripts/build-wheel.sh` contains "/build"; METADATA carries the README."""
    tree = tmp_path / "tree"
    (tree / "dolfinx_solver").mkdir(parents=True)
    (tree / "dolfinx_solver" / "__init__.py").write_text(
        "# built by scripts/build-wheel.sh\n"
    )
    (tree / "dolfinx_solver" / "leak.cfg").write_text("PETSC_DIR = /build/install\n")

    built = assemble.observe(
        assemble.pack(tree, tmp_path / "w.whl"),
        tmp_path / "unpacked",
        build_root=Path("/build"),
    )

    assert built.build_paths == ("dolfinx_solver/leak.cfg",)


def test_an_unpruned_transport_dependency_is_refused():
    needed = dict(_good_wheel().needed)
    needed[f"{LIBRARY_DIR}/{mpich.FORTRAN_SONAME}"] = frozenset(
        {"libmpi.so.12", "libucp.so.0"}
    )
    problem = assemble.wheel_problem(_good_wheel(needed=needed))

    assert problem is not None
    assert "libucp.so.0" in problem


def test_the_import_check_runs_against_the_installed_site():
    """The acceptance criterion: it imports out of a clean venv, not a staging tree."""
    arguments = assemble.import_check_arguments(
        Path("/venv/lib/python3.12/site-packages"), python="/venv/bin/python"
    )

    assert arguments[:3] == ["/venv/bin/python", "-m", "wheelbuild.import_check"]
    assert "--dolfinx" in arguments
    assert "/venv/lib/python3.12/site-packages" in arguments


def test_the_wheel_is_installed_on_its_own_metadata():
    """pip resolves mpich and the trio from the index; that is the test."""
    arguments = assemble.install_arguments(
        Path("/wheelhouse/w.whl"), python="/venv/bin/python"
    )

    assert "--no-deps" not in arguments
    assert arguments[-1] == "/wheelhouse/w.whl"


def test_the_distribution_name_is_the_variant_the_drivers_build():
    """Spec §6: the scalar type is part of the name, and one place declares
    it. The notice file spells the same name as PyPI does (ticket 22)."""
    assert f"dolfinx_solver_{petsc.SCALAR_TYPE}" == assemble.DISTRIBUTION
    assert notices.DISTRIBUTION.replace("-", "_") == assemble.DISTRIBUTION


@pytest.mark.parametrize("variant", petsc.SCALAR_TYPES)
def test_the_copied_metadata_names_the_variant_being_built(variant):
    """The base wheel is built from this text, so it is what decides the
    distribution a user installs and the name the payload reads back."""
    retargeted = assemble.retarget_project(PROJECT_METADATA, scalar_type=variant)

    assert f'name = "dolfinx-solver-{variant}"' in retargeted
    assert f"with {variant}-scalar PETSc" in retargeted


def test_the_checked_in_metadata_is_the_default_variants_own():
    """Which is why a copy for that variant comes out byte-identical, and why
    `pip install -e .` in this checkout installs the name CI publishes."""
    checked_in = (assemble.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        assemble.retarget_project(checked_in, scalar_type=petsc.DEFAULT_SCALAR_TYPE)
        == checked_in
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        'name = "dolfinx-solver"\n',
        PROJECT_METADATA.replace("description", "summary"),
        PROJECT_METADATA * 2,
    ],
)
def test_metadata_the_substitution_cannot_place_fails_the_build(text):
    """A silent miss would publish a wheel named for the other variant."""
    with pytest.raises(ValueError, match=r"pyproject\.toml"):
        assemble.retarget_project(text, scalar_type="real")


def test_the_source_copy_retargets_the_metadata_it_copies(tmp_path):
    """One of the two files in SOURCE_INPUTS that are not copied verbatim."""
    repo = _repository(tmp_path / "repo")

    copied = assemble.source_copy(tmp_path / "copy", repo_root=repo)

    assert f'name = "dolfinx-solver-{petsc.SCALAR_TYPE}"' in (
        copied / "pyproject.toml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("variant", petsc.SCALAR_TYPES)
def test_the_copied_readme_describes_the_variant_being_built(variant):
    """It becomes the wheel's `Description`, which is the project page: a
    verbatim copy would sell the real wheel as a complex-scalar build and
    print the other distribution's install line under it (ticket 32)."""
    other = next(name for name in petsc.SCALAR_TYPES if name != variant)

    retargeted = assemble.retarget_readme(README, scalar_type=variant)

    assert f"**{variant}-scalar PETSc and SLEPc inside the wheel**" in retargeted
    assert f"{other}-scalar PETSc and SLEPc inside the wheel" not in retargeted
    assert f"pip install dolfinx-solver-{variant}" in retargeted
    assert f"pip install dolfinx-solver-{other}" not in retargeted
    assert f"published separately as `dolfinx-solver-{other}`" in retargeted


@pytest.mark.parametrize("variant", petsc.SCALAR_TYPES)
def test_the_copied_readme_shows_the_scalar_type_the_wheel_has(variant):
    """The snippet is what a reader runs first; the complex one's assertion
    raises a TypeError on a real build."""
    expected = {"complex": "complex128", "real": "float64"}[variant]

    retargeted = assemble.retarget_readme(README, scalar_type=variant)

    assert f'assert PETSc.ScalarType.__name__ == "{expected}"' in retargeted


def test_the_checked_in_readme_is_the_default_variants_own():
    """The same rule the metadata follows: a copy for the declared variant
    comes out byte-identical, so GitHub's README and the complex wheel's
    project page are the same text."""
    checked_in = (assemble.REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        assemble.retarget_readme(checked_in, scalar_type=petsc.DEFAULT_SCALAR_TYPE)
        == checked_in
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "pip install dolfinx-solver\n",
        README.replace("```console", "```bash"),
        README * 2,
    ],
)
def test_a_readme_the_substitutions_cannot_place_fails_the_build(text):
    """A silent miss would publish a wheel whose page describes the other
    variant, which is the failure this file exists to prevent."""
    with pytest.raises(ValueError, match=r"README\.md"):
        assemble.retarget_readme(text, scalar_type="real")


def test_the_source_copy_retargets_the_readme_it_copies(tmp_path):
    """The second file in SOURCE_INPUTS that is not copied verbatim."""
    repo = _repository(tmp_path / "repo")

    copied = assemble.source_copy(tmp_path / "copy", repo_root=repo)

    assert f"pip install dolfinx-solver-{petsc.SCALAR_TYPE}" in (
        copied / "README.md"
    ).read_text(encoding="utf-8")
