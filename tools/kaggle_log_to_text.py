"""
Convert a Kaggle notebook log into plain text for logs/.

Kaggle stores a run's log as a JSON list of {"stream_name", "time", "data"} entries, where "time" is
seconds since the run started and "data" is a chunk of stdout or stderr. This writes every chunk, in
order, as lines prefixed with the elapsed time and the stream, so nothing is dropped and the training
progress can be read in any text viewer.

Usage: python3 tools/kaggle_log_to_text.py <kaggle .log file> logs/<notebook>_v<N>.txt
"""
import json, sys


def convert(src, dst):
    """Read the Kaggle JSON log at src and write the readable text version to dst."""
    entries = json.load(open(src))
    with open(dst, "w") as out:
        for e in entries:
            prefix = f"[{e['time']:8.1f}s {e['stream_name']}] "      # elapsed seconds since the run started
            for line in e["data"].rstrip("\n").split("\n"):
                out.write(prefix + line + "\n")
    print(f"{len(entries)} log entries -> {dst}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
