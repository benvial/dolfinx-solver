"""What two ranks have to agree on, judged without two ranks."""

from wheeltest import interop


def test_two_ranks_in_one_communicator_pass():
    assert interop.size_problem(2) is None


def test_a_launcher_that_started_two_single_rank_jobs_fails():
    """The failure mode that otherwise runs the whole suite twice and passes."""
    problem = interop.size_problem(1)

    assert problem is not None
    assert "hydra" in problem


def test_an_evenly_partitioned_mesh_passes():
    assert interop.partition_problem([256, 256], 512) is None


def test_a_mesh_one_rank_owns_entirely_fails():
    problem = interop.partition_problem([512, 0], 512)

    assert problem is not None
    assert "PT-SCOTCH" in problem


def test_ranks_that_built_different_meshes_fail():
    """Owned cells partition the mesh; these ranks are not sharing one."""
    problem = interop.partition_problem([512, 512], 512)

    assert problem is not None
    assert "own 1024 cells between them" in problem


def test_an_uneven_but_real_partition_passes():
    assert interop.partition_problem([300, 212], 512) is None


def test_two_reductions_of_one_vector_agreeing_passes():
    assert interop.norm_problem(27.073973, 27.073973 + 1e-13) is None


def test_two_reductions_of_one_vector_disagreeing_fails():
    problem = interop.norm_problem(27.073973, 19.14)

    assert problem is not None
    assert "not the one mpi4py is reducing over" in problem


def test_a_petsc_vector_on_a_smaller_communicator_fails():
    facts = {
        "size": 2,
        "petsc_comm_size": 1,
        "local_cells": [256, 256],
        "global_cells": 512,
        "petsc_norm": 1.0,
        "mpi4py_norm": 1.0,
    }

    problem = interop.facts_problem(facts)

    assert problem is not None
    assert "not holding the same MPI" in problem


def test_a_run_where_everything_agrees_passes():
    facts = {
        "size": 2,
        "petsc_comm_size": 2,
        "local_cells": [256, 256],
        "global_cells": 512,
        "petsc_norm": 27.073973,
        "mpi4py_norm": 27.073973,
    }

    assert interop.facts_problem(facts) is None
