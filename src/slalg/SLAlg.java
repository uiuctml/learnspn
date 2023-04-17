package slalg;

import spn.GraphSPN;
import data.Dataset;

public interface SLAlg {
	public GraphSPN learnStructure(Dataset d); 
}
