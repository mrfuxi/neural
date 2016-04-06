package neural

import "io"

// SaverLoader define persisting network and loading previously persisted data
type SaverLoader interface {
	Save(w io.Writer) error
	Load(r io.Reader) error
}

// Save persists trained network into a writer
func Save(nn Evaluator, w io.Writer) error {
	for _, layer := range nn.Layers() {
		if err := layer.Save(w); err != nil {
			return err
		}
	}
	return nil
}

// Load using reader to restore previously persisted data into configured network.
// Network has to have correct shape when loading data
func Load(nn Evaluator, r io.Reader) error {
	for _, layer := range nn.Layers() {
		if err := layer.Load(r); err != nil {
			return err
		}
	}
	return nil
}
