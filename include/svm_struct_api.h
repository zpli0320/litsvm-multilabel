/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.h                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#ifndef svm_struct_api
#define svm_struct_api

#include "svm_struct_api_types.h"
#include "svm_struct_common.h"
#include <typeinfo>
#include <bitset>

using std::bitset;

void        svm_struct_learn_api_init(int argc, char* argv[]);
void        svm_struct_learn_api_exit();
void        svm_struct_classify_api_init(int argc, char* argv[]);
void        svm_struct_classify_api_exit();
SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm);
void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm);
CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm);
int         empty_label(LABEL y);
SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm, 
	        STRUCT_LEARN_PARM *sparm);
double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm);
void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm);
void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm,
				       STRUCT_TEST_STATS *teststats);
void        eval_prediction(long exnum, EXAMPLE ex, LABEL prediction, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
			    STRUCT_TEST_STATS *teststats);
void        write_struct_model(char *file,STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm);
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm);
void        write_label(FILE *fp, LABEL y);
void        free_pattern(PATTERN x);
void        free_label(LABEL y);
void        free_struct_model(STRUCTMODEL sm);
void        free_struct_sample(SAMPLE s);
void        print_struct_help();
void        parse_struct_parameters(STRUCT_LEARN_PARM *sparm);
void        print_struct_help_classify();
void        parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm);
void        svm_learn_struct_joint_custom(SAMPLE sample, 
				   STRUCT_LEARN_PARM *sparm,
				   LEARN_PARM *lparm, KERNEL_PARM *kparm, 
				   STRUCTMODEL *sm);

void  read_instance_multilabel(char *line, PATTERN &x, LABEL &y);
template <typename Value> Value read_num(char* line, long &pos){
Value number;
if(typeid(number) == typeid(int)) {
sscanf(line + pos, "%d", &number);
}
else if(typeid(number) == typeid(double)) {
sscanf(line + pos, "%lf", &number);
}

while(line[pos] != ' ' && line[pos] !=':' && line[pos]) pos++;
while(line[pos] ==' ' || line[pos] == ':') pos++;
return number;
}
int read_label (char *line, long &pos);
void b2v(bitset<32> &bitvec, vector<int> &y, int len);
int diff_label_num(vector<int> &y_truth, vector<int> &y_predict);
#endif
