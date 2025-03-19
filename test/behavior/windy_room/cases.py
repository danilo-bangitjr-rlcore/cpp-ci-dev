from test.behavior.bsuite import BSuiteTestCase


class WindyRoomTest(BSuiteTestCase):
    name = 'windy_room'
    config = 'test/behavior/windy_room/windy_room.yaml'

    upper_bounds = {'red_zone_violation': 0.}
    lower_goals = {'yellow_zone_violation': 0.}

    aggregators = {'red_zone_violation': 'percent_of_steps',
                   'yellow_zone_violation': 'percent_of_steps',
                   }
