#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import six
import re
import threading
import unittest


# def synchronized_method(method):
#     outer_lock = threading.Lock()
#     lock_name = "__" + method.__name__ + "_lock" + "__"
#
#     def sync_method(self, *args, **kws):
#         with outer_lock:
#             if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
#             lock = getattr(self, lock_name)
#             with lock:
#                 return method(self, *args, **kws)
#
#     return sync_method


class SubProcess(object):
    def __init__(self, command):
        self._lock = threading.Lock()
        subproc_args = {'stdin': subprocess.PIPE, 'stdout': subprocess.PIPE,
                        'stderr': subprocess.STDOUT, 'cwd': '.',
                        'close_fds': sys.platform != "win32"}
        try:
            env = os.environ.copy()
            self.process = subprocess.Popen('bash -c "%s"' % command, env=env,
                                            shell=True, **subproc_args)
        except OSError:
            raise
        (self.stdouterr, self.stdin) = (self.process.stdout, self.process.stdin)

    def __del__(self):
        self.process.stdin.close()
        try:
            self.process.kill()
            self.process.wait()
        except OSError:
            pass

    def query(self, sentence, pattern):
        assert (isinstance(sentence, six.text_type))
        with self._lock:
            self.process.stdin.write(sentence.encode('utf-8') + six.b('\n'))
            self.process.stdin.flush()
            result = ""
            while True:
                line = self.stdouterr.readline()[:-1].decode('utf-8')
                if re.search(pattern, line):
                    break
                result = "%s%s\n" % (result, line)
        return result


class SubProcessTest(unittest.TestCase):
    def test(self):
        self.juman_process = SubProcess("juman")
        self.pattern = r'EOS'
        self.juman_process.query(u"オバマは第４４代アメリカ大統領である。", self.pattern)

    def test_multithread(self):
        self.juman_process = SubProcess("juman")
        self.pattern = r'EOS'
        threads = []
        for i in range(100):
            args = (u"オバマは第４４代アメリカ大統領である。", self.pattern)
            thread = threading.Thread(target=self.juman_process.query, args=args)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


if __name__ == '__main__':
    unittest.main()
