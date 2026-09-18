"""Compatibilidade: sem argumentos abre o dashboard; com argumentos usa a CLI."""
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        from run import main
    else:
        from safeguard.cli import main
    raise SystemExit(main())
