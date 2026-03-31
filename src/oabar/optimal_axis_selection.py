from src.oabar.obstacle_aware_divider import ObstacleAwareDivider
from src.oabar.strip_perimeter import Strip


class OptimalAxisSelection:
    """
    Evaluates NWCRT on x/y and uses MSDU as tie-breaker.
    Minimal robustness: axis-level try/except + symmetric squareness.
    """

    def __init__(self, region, obstacles, user_metric=None, numerical_method='newton',
                 tie_threshold=1e-2):
        self.region = region
        self.obstacles = obstacles
        self.numerical_method = numerical_method
        self.tie_threshold = tie_threshold
        self.user_metric = user_metric if user_metric == "NWCRT" else "NWCRT"

    def evaluate_axis(self, axis):
        strip_manager = Strip(self.region, self.obstacles, axis=axis)
        divider = ObstacleAwareDivider(strip_manager, method=self.numerical_method)

        division_point = divider.find_optimal_division_point()
        (R_left, left_obstacles), (R_right, right_obstacles) = divider.divide_region(division_point)

        left_sm = Strip(R_left, left_obstacles, axis=axis)
        right_sm = Strip(R_right, right_obstacles, axis=axis)

        wcrt_left = left_sm.calculate_region_wcrt()
        wcrt_right = right_sm.calculate_region_wcrt()

        sum_wcrt = wcrt_left + wcrt_right
        diff_wcrt = abs(wcrt_left - wcrt_right)

        metrics = {}
        metrics["NWCRT"] = diff_wcrt / sum_wcrt if sum_wcrt > 1e-9 else float("inf")

        # Tie-breaker (MSDU):
        sq_left = self._square_measure(R_left)
        sq_right = self._square_measure(R_right)
        _, msdu = self.calculate_squareness_metrics(sq_left, sq_right)
        metrics["_MSDU"] = msdu

        metrics["_division_point"] = division_point
        metrics["_subregion_left"] = (R_left, left_obstacles)
        metrics["_subregion_right"] = (R_right, right_obstacles)
        return metrics

    def _square_measure(self, polygon, eps=1e-9):
        """
        Symmetric squareness in (0,1]: min(w/h, h/w).
        1.0 means perfect square; smaller => more skinny.
        """
        if polygon.is_empty:
            return 1.0
        minx, miny, maxx, maxy = polygon.bounds
        w = maxx - minx
        h = maxy - miny
        if w < eps or h < eps:
            return 0.0
        r1 = w / h
        r2 = h / w
        return min(r1, r2)

    def calculate_squareness_metrics(self, sq_left, sq_right, eps=1e-9):
        msdu = 1.0 / ((0.5 * ((sq_left - 1.0) ** 2 + (sq_right - 1.0) ** 2)) + eps)
        return None, msdu

    def select_best_axis(self):
        """
        Robust: if one axis fails, pick the other.
        If both fail, raise (caller will track_back).
        """
        metrics_x = None
        metrics_y = None
        err_x = None
        err_y = None

        try:
            metrics_x = self.evaluate_axis('x')
        except Exception as e:
            err_x = e

        try:
            metrics_y = self.evaluate_axis('y')
        except Exception as e:
            err_y = e

        if metrics_x is None and metrics_y is None:
            raise RuntimeError(f"Both axes failed. x_err={err_x}, y_err={err_y}")

        if metrics_x is None:
            chosen_metrics = metrics_y
            best_axis = 'y'
        elif metrics_y is None:
            chosen_metrics = metrics_x
            best_axis = 'x'
        else:
            value_x = metrics_x["NWCRT"]
            value_y = metrics_y["NWCRT"]

            diff = abs(value_x - value_y)
            if diff <= self.tie_threshold:
                best_axis = 'x' if metrics_x["_MSDU"] > metrics_y["_MSDU"] else 'y'
                chosen_metrics = metrics_x if best_axis == 'x' else metrics_y
            else:
                best_axis = 'x' if value_x < value_y else 'y'
                chosen_metrics = metrics_x if best_axis == 'x' else metrics_y

        best_div_point = chosen_metrics["_division_point"]
        best_sub_left = chosen_metrics["_subregion_left"]
        best_sub_right = chosen_metrics["_subregion_right"]

        overall_metrics = {"x": metrics_x, "y": metrics_y}
        return best_axis, overall_metrics, best_div_point, best_sub_left, best_sub_right
