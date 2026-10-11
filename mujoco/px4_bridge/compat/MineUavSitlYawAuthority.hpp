// Local opt-in SITL compatibility guard. NOT an upstream/hardware controller.
// Matrix entries and the [0,1] actuator box are never rescaled.
#pragma once
#include <cmath>

// Passive compatibility eligibility, independent of CA_FAILURE_MODE handling.
// Unknown detector state is fail-closed. A transition requests matrix rebuild.
struct MineUavSitlMotorHealth {
	bool known {false};
	bool fault {true};
	bool allowed() const { return known && !fault; }
	bool observe(bool detected, unsigned failure_mask, unsigned stop_mask) {
		const bool previous = allowed();
		known = true;
		fault = detected || failure_mask != 0u || stop_mask != 0u;
		return previous != allowed();
	}
};

template<typename Matrix>
bool mineuavSitlYawAuthority(const Matrix &matrix, int count)
{
	if (count != 4) { return false; }
	float yaw_min = 1.f, yaw_max = -1.f;
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 6; ++row) {
			if (!std::isfinite(matrix(row, col))) { return false; }
		}
		// Restricted to the validated non-tilting CT=1 normalized motor model.
		if (std::fabs(matrix(3, col)) > 1e-6f || std::fabs(matrix(4, col)) > 1e-6f
		    || std::fabs(matrix(5, col) + 1.f) > 1e-6f) { return false; }
		if (matrix(2, col) < yaw_min) { yaw_min = matrix(2, col); }
		if (matrix(2, col) > yaw_max) { yaw_max = matrix(2, col); }
	}
	if (yaw_min > -.01f || yaw_max < .01f || yaw_min < -.05f || yaw_max > .05f) { return false; }
	// RP must survive the unchanged legacy filter as well: do not certify
	// independence against rows that ControlAllocator will subsequently erase.
	for (int row = 0; row < 2; ++row) {
		float maximum = 0.f;
		for (int col = 0; col < 4; ++col) {
			if (std::fabs(matrix(row, col)) > maximum) { maximum = std::fabs(matrix(row, col)); }
		}
		if (maximum <= .05f) { return false; }
	}
	const int axes[4] {5, 0, 1, 2};
	double basis[4][4] {}, system[4][5] {};
	for (int row = 0; row < 4; ++row) {
		double norm2 = 0.;
		for (int col = 0; col < 4; ++col) { norm2 += double(matrix(axes[row], col)) * double(matrix(axes[row], col)); }
		if (norm2 < 1e-6) { return false; }
		const double norm = std::sqrt(norm2);
		for (int col = 0; col < 4; ++col) {
			basis[row][col] = double(matrix(axes[row], col)) / norm;
			system[row][col] = basis[row][col];
		}
		system[row][4] = row == 0 ? -1. / norm : 0.;
		// Twice-modified Gram-Schmidt: independent, well-separated rows only.
		for (int pass = 0; pass < 2; ++pass) {
			for (int earlier = 0; earlier < row; ++earlier) {
				double dot = 0.;
				for (int col = 0; col < 4; ++col) { dot += basis[row][col] * basis[earlier][col]; }
				for (int col = 0; col < 4; ++col) { basis[row][col] -= dot * basis[earlier][col]; }
			}
		}
		double residual2 = 0.;
		for (int col = 0; col < 4; ++col) { residual2 += basis[row][col] * basis[row][col]; }
		if (residual2 < .25) { return false; }
		for (int col = 0; col < 4; ++col) { basis[row][col] /= std::sqrt(residual2); }
	}
	// Zero-moment unit-collective solution must be interior to the motor box.
	// Rank alone would not guarantee feasible trim for nonreversible motors.
	for (int pivot = 0; pivot < 4; ++pivot) {
		int best = pivot;
		for (int row = pivot + 1; row < 4; ++row) {
			if (std::fabs(system[row][pivot]) > std::fabs(system[best][pivot])) { best = row; }
		}
		for (int col = 0; col < 5; ++col) {
			const double tmp = system[pivot][col]; system[pivot][col] = system[best][col]; system[best][col] = tmp;
		}
		const double divisor = system[pivot][pivot];
		if (std::fabs(divisor) < 1e-6) { return false; }
		for (int col = 0; col < 5; ++col) { system[pivot][col] /= divisor; }
		for (int row = 0; row < 4; ++row) {
			if (row == pivot) { continue; }
			const double factor = system[row][pivot];
			for (int col = 0; col < 5; ++col) { system[row][col] -= factor * system[pivot][col]; }
		}
	}
	for (int row = 0; row < 4; ++row) {
		if (!std::isfinite(system[row][4]) || system[row][4] <= .1 || system[row][4] >= .4) { return false; }
	}
	return true;
}
