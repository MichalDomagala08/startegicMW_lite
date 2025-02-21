import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from storyTrial import StoryTrial
from psychopy import core

class TestStoryTrial(unittest.TestCase):
    
    """
        Failed Attemptc at Testing Things - good try, But Unit Testing by the Class works only, when the precise and real timing is not involved
        Better to use functions 
    
    """
    def setUp(self):
        self.story_duration = 5  # Short test duration
        self.mock_story_sound = MagicMock()
        self.mock_story_sound.getDuration.return_value = self.story_duration
        self.test_trial = StoryTrial(story_sound=self.mock_story_sound, TP_dist=(1, 2), verbose=1)
        self.tolerance = 0.005

    @patch('random.uniform', return_value=1.5)  # Thought probes every 1.5s
    @patch('psychopy.core.wait', side_effect=lambda x: core.wait(0.01))  # Speed up the waiting
    def test_Absolute_RT_time(self, mock_wait, mock_uniform):
        initial_time = core.getTime()

        # Correct keypress mock
        fake_keypress_a = MagicMock(name='KeyEvent', spec_set=['name'])
        fake_keypress_a.name = 'a'

        # Define when keypresses should occur
        keypress_times = [1.6, 3.1]
        current_keypress = iter(keypress_times)

        def keypress_generator(*args, **kwargs):
            """ Return keypress events only when current time matches expected keypress times. """
            current_time = core.getTime() - initial_time
            next_keypress = next(current_keypress, None)
            if next_keypress is not None and abs(current_time - next_keypress) < 0.1:
                return [fake_keypress_a]
            return []  # No keypress

        fake_kb = MagicMock()
        fake_kb.getKeys.side_effect = keypress_generator

        # Inject the mocked keyboard into the trial
        self.test_trial.kb = fake_kb

        # Run the trial
        self.test_trial.run()

        # Get the log list and check the results
        loglist = self.test_trial.get_logList()
        print("LogList contents:", pd.DataFrame(loglist))

        for i in range(len(loglist) - 1):
            if loglist[i][0] == "PROBE" and loglist[i + 1][0] == "KEYPRESS":
                probe_time = loglist[i][1]
                keypress_time = loglist[i + 1][1]
                reaction_time = loglist[i + 1][3]

                print(f"\nChecking PROBE at {probe_time:.3f}s and KEYPRESS at {keypress_time:.3f}s")
                print(f"Expected KEYPRESS: {probe_time + reaction_time:.3f}s, Actual: {keypress_time:.3f}s")

                self.assertAlmostEqual(probe_time + reaction_time, keypress_time, delta=self.tolerance,
                                       msg=f"Absolute Timing mismatch for PROBE-KEYPRESS pair at row {i}")

if __name__ == '__main__':
    unittest.main()


def testTiming(loglist):
    print("*** Testing Absolute Timing ***  ")
    for i in range(len(loglist) - 1):
        if loglist[i][0] == "PROBE" and loglist[i + 1][0] == "KEYPRESS":
            probe_time = loglist[i][1]
            keypress_time = loglist[i + 1][1]
            reaction_time = loglist[i + 1][3]
            print(f"Row: {i:5.0f}; RT: {reaction_time:2.3f}; KeyPress: {keypress_time:15.3f}; probe_time:  {probe_time:15.3f}")

            if abs((probe_time + reaction_time) - keypress_time) <0.005:
                continue
            else:
                raise ValueError(f'Mismatch at {i}th event at time {probe_time}')

    print("*** Testing Session Timing ***  ")
    for i in range(len(loglist) - 1):
        if loglist[i][0] == "PROBE" and loglist[i + 1][0] == "KEYPRESS":
            probe_time = loglist[i][2]
            keypress_time = loglist[i + 1][2]
            reaction_time = loglist[i + 1][3]
            print(f"Row: {i:5.0f}; RT: {reaction_time:2.3f}; KeyPress: {keypress_time:15.3f}; probe_time:  {probe_time:15.3f}")

            if abs((probe_time + reaction_time) - keypress_time) <0.005:
                continue
            else:
                raise ValueError(f'Mismatch at {i}th event at time {probe_time}')


    print("*** Testing Probe Timing ***  ")
    probeList = [i for i in loglist if i[0] =="PROBE"];
    for i in range(len(probeList) - 1):
        probe_time = probeList[i][2]
        interval_time = probeList[i + 1][4]
        future_probe  = probeList[i + 1][2]
        print(f"Row: {i:5.0f}; IntervalTime: {interval_time:2.3f}; future_probe: {future_probe:15.3f}; probe_time:  {probe_time:15.3f}")

        if abs(future_probe - probe_time - interval_time) <0.005:
            continue
        else:
            raise ValueError(f'Mismatch at {i+1}th event at time {probe_time}')

