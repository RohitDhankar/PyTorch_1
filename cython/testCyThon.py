
import timeit
#import fromScratch

cythonCode = timeit.timeit('fromScratch_cy.Conv_class(18,7)',setup = 'import fromScratch_cy', number = 1)
pythonCode = timeit.timeit('fromScratch.Conv_class(18,7)',setup = 'import fromScratch', number = 1)

print(cythonCode,pythonCode)
print('CyThon = cythoCode, executed {}times faster than Python Code'.format(pythonCode/cythonCode))
# mostly 6.5 Times faster
# example_cy.testCyThon(50)==  22 Times 
# example_cy.testCyThon(500) == 138.66 Times faster


